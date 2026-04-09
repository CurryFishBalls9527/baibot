"""Monitor selected X accounts via RSS mirrors and push bilingual alerts."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional
from urllib import parse, request

import feedparser
from openai import OpenAI

from .notifier import NtfyNotifier

logger = logging.getLogger(__name__)


class SocialFeedMonitor:
    """Fetch RSS feeds for selected X accounts and notify on new posts."""

    def __init__(self, config: dict):
        self.config = config
        self.enabled = bool(config.get("social_monitor_enabled", False))
        raw_usernames = config.get("social_monitor_usernames", [])
        if isinstance(raw_usernames, str):
            raw_usernames = [item.strip() for item in raw_usernames.split(",") if item.strip()]
        self.usernames = [item.lstrip("@").strip() for item in raw_usernames if item.strip()]
        self.feed_url_template = str(
            config.get("social_feed_url_template", "https://nitter.net/{username}/rss")
        )
        self.x_bearer_token = os.getenv("X_BEARER_TOKEN", "").strip()
        self.x_api_base = "https://api.x.com/2"
        self.notifier = NtfyNotifier(
            {
                "ntfy_enabled": config.get("social_ntfy_enabled", False),
                "ntfy_server": config.get(
                    "social_ntfy_server",
                    config.get("ntfy_server", "https://ntfy.sh"),
                ),
                "ntfy_topic": config.get("social_ntfy_topic", ""),
                "ntfy_priority": config.get("social_ntfy_priority", "default"),
                "ntfy_tags": config.get("social_ntfy_tags", ["newspaper", "speech_balloon"]),
                "ntfy_click_url": config.get("social_ntfy_click_url", ""),
                "results_dir": config.get("results_dir", "./results"),
            }
        )
        self.model = str(
            config.get("social_translation_model", config.get("quick_think_llm", "gpt-4o-mini"))
        )
        self.openai_client: Optional[OpenAI] = None
        if os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.state_path = (
            Path(config.get("results_dir", "./results"))
            / "social_monitor"
            / "social_state.json"
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def check_once(self) -> Dict:
        if not self.enabled or not self.usernames:
            return {"enabled": False, "users": 0, "alerts": 0}

        state = self._load_state()
        total_alerts = 0
        updated = False
        account_results = {}

        for username in self.usernames:
            entries, source = self._fetch_entries(username)
            if not entries:
                account_results[username] = {
                    "fetched": 0,
                    "alerts": 0,
                    "bootstrapped": False,
                    "source": source,
                }
                continue

            newest_id = self._entry_id(entries[0])
            if not newest_id:
                account_results[username] = {
                    "fetched": len(entries),
                    "alerts": 0,
                    "bootstrapped": False,
                    "source": source,
                }
                continue

            account_state = state.get(username, {})
            last_seen_id = account_state.get("last_seen_id")
            if not last_seen_id:
                state[username] = {
                    "last_seen_id": newest_id,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                updated = True
                account_results[username] = {
                    "fetched": len(entries),
                    "alerts": 0,
                    "bootstrapped": True,
                    "source": source,
                }
                continue

            new_entries = []
            for entry in entries:
                entry_id = self._entry_id(entry)
                if not entry_id:
                    continue
                if entry_id == last_seen_id:
                    break
                new_entries.append(entry)

            if new_entries:
                for entry in reversed(new_entries):
                    if self._notify_entry(username, entry):
                        total_alerts += 1
                state[username] = {
                    "last_seen_id": newest_id,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                updated = True

            account_results[username] = {
                "fetched": len(entries),
                "alerts": len(new_entries),
                "bootstrapped": False,
                "source": source,
            }

        if updated:
            self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        return {
            "enabled": True,
            "users": len(self.usernames),
            "alerts": total_alerts,
            "accounts": account_results,
            "topic": self.notifier.topic if self.notifier.enabled else None,
        }

    def send_test_notification(self) -> bool:
        if not self.notifier.enabled:
            return False
        message = "\n".join(
            [
                "Account: @markminervini",
                "English: Test social alert. New X post monitoring is connected and ready.",
                "中文: 这是社交监控测试通知。新的 X 帖子监控已经连接完成。",
                "Source: simulated social monitor event",
            ]
        )
        return self.notifier.send(
            "Social Monitor Test",
            message,
            priority="high",
            tags=["newspaper", "iphone"],
        )

    def _notify_entry(self, username: str, entry) -> bool:
        english, chinese = self._bilingual_summary(entry)
        message = "\n".join(
            [
                f"Account: @{username}",
                f"English: {english}",
                f"中文: {chinese}",
            ]
        )
        return self.notifier.send(
            f"X Alert: @{username}",
            message,
            priority="high",
            tags=["newspaper", "speech_balloon"],
            dedupe_key=f"social:{username}:{self._entry_id(entry)}",
        )

    def _bilingual_summary(self, entry) -> tuple[str, str]:
        title = self._clean_text(self._entry_value(entry, "title") or "")
        summary = self._clean_text(self._entry_value(entry, "summary") or "")
        text = summary or title or "New post detected."
        if self.openai_client is None:
            return text[:220], f"检测到新帖子：{text[:120]}"

        prompt = (
            "Convert the following X post into a concise bilingual alert. "
            "Return strict JSON with keys english and chinese. "
            "Keep each field under 180 characters. "
            "Preserve tickers and important trading language.\n\n"
            f"Title: {title}\n"
            f"Post: {text}"
        )
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You produce concise bilingual market alerts in JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content or "{}"
            payload = json.loads(raw)
            english = self._clean_text(str(payload.get("english", "") or text))[:220]
            chinese = self._clean_text(
                str(payload.get("chinese", "") or f"检测到新帖子：{text}")
            )[:220]
            return english, chinese
        except Exception as exc:
            logger.warning("Social translation failed: %s", exc)
            return text[:220], f"检测到新帖子：{text[:120]}"

    def _fetch_entries(self, username: str) -> tuple[List, str]:
        if self.x_bearer_token:
            entries = self._fetch_entries_official(username)
            if entries:
                return entries, "x_api"
        entries = self._fetch_entries_rss(username)
        return entries, "rss"

    def _fetch_entries_official(self, username: str) -> List[Dict]:
        user_id = self._fetch_user_id(username)
        if not user_id:
            return []
        params = parse.urlencode(
            {
                "max_results": 10,
                "exclude": "replies,retweets",
                "tweet.fields": "created_at,text",
            }
        )
        url = f"{self.x_api_base}/users/{user_id}/tweets?{params}"
        payload = self._request_json(
            url,
            headers={"Authorization": f"Bearer {self.x_bearer_token}"},
        )
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        entries = []
        for row in rows:
            post_id = str(row.get("id", "")).strip()
            text = self._clean_text(str(row.get("text", "") or ""))
            if not post_id or not text:
                continue
            entries.append(
                {
                    "id": post_id,
                    "title": text,
                    "summary": text,
                    "link": f"https://x.com/{username}/status/{post_id}",
                    "created_at": row.get("created_at"),
                }
            )
        return entries

    def _fetch_user_id(self, username: str) -> Optional[str]:
        url = f"{self.x_api_base}/users/by/username/{parse.quote(username)}"
        payload = self._request_json(
            url,
            headers={"Authorization": f"Bearer {self.x_bearer_token}"},
        )
        if not isinstance(payload, dict):
            return None
        data = payload.get("data") or {}
        user_id = data.get("id")
        return str(user_id) if user_id else None

    def _fetch_entries_rss(self, username: str) -> List:
        feed_url = self.feed_url_template.format(username=username)
        req = request.Request(
            feed_url,
            headers={"User-Agent": "Mozilla/5.0 TradingAgents/1.0"},
        )
        try:
            with request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
            parsed = feedparser.parse(raw)
            return list(parsed.entries[:10])
        except Exception as exc:
            logger.warning("Social feed fetch failed for @%s: %s", username, exc)
            return []

    def _request_json(self, url: str, headers: Optional[dict] = None) -> Dict:
        req_headers = {"User-Agent": "Mozilla/5.0 TradingAgents/1.0"}
        if headers:
            req_headers.update(headers)
        req = request.Request(url, headers=req_headers)
        try:
            with request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Social API request failed for %s: %s", url, exc)
            return {}

    @staticmethod
    def _clean_text(text: str) -> str:
        text = unescape(text or "")
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _entry_id(entry) -> str:
        for key in ("id", "guid", "link", "title"):
            value = SocialFeedMonitor._entry_value(entry, key)
            if value:
                return str(value)
        return ""

    @staticmethod
    def _entry_value(entry, key: str):
        if isinstance(entry, dict):
            return entry.get(key)
        return getattr(entry, key, None)

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
