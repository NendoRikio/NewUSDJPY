"""
Forex news fetcher and sentiment analysis module.
Retrieves latest FX, Fed, and BoJ news via Google News RSS and analyzes market sentiment.
"""
from datetime import datetime
import re
import feedparser


# Keywords defining USD/JPY bullish (USD up / JPY down)
BULLISH_KEYWORDS_USD = [
    'ドル高', '円安', 'ドル買い', '円売り', '米金利上昇', '利上げ', 'タカ派', 'hawkish',
    'インフレ加速', 'インフレ高進', '米景気好調', '米雇用強い', 'パウエル利上げ', '利下げ後退',
    '介入見送り', 'ドル堅調', '上昇基調', '高値更新', 'usd strong', 'dollar surge',
    'rate hike', 'higher for longer', 'inflation sticky'
]

# Keywords defining USD/JPY bearish (USD down / JPY up)
BEARISH_KEYWORDS_USD = [
    'ドル安', '円高', 'ドル売り', '円買い', '米金利低下', '利下げ', 'ハト派', 'dovish',
    '景気減速', '景気後退', '米雇用悪化', '日銀利上げ', '為替介入', '口先介入', '植田利上げ',
    '利下げ加速', '下落基調', '安値更新', 'usd weak', 'yen rally', 'rate cut',
    'intervention', 'boj hike', 'recession fears'
]


def fetch_latest_fx_news(limit: int = 15) -> tuple[list[dict], dict]:
    """
    Fetches latest FX news and scores USD/JPY sentiment.
    Returns: (list of news items, sentiment summary dict)
    """
    feeds = [
        # Japanese FX news
        'https://news.google.com/rss/search?q=USD+JPY+OR+ドル円+OR+為替&hl=ja&gl=JP&ceid=JP:ja',
        # English FX / Fed news
        'https://news.google.com/rss/search?q=USD+JPY+forex+Fed+OR+Bank+of+Japan&hl=en-US&gl=US&ceid=US:en'
    ]

    news_items = []
    seen_titles = set()

    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                title = entry.get('title', '').strip()
                if not title:
                    continue

                # Deduplicate by simplified title
                clean_title = re.sub(r'\s+', ' ', title).lower()
                if clean_title in seen_titles:
                    continue
                seen_titles.add(clean_title)

                link = entry.get('link', '')
                published = entry.get('published', '')
                source = entry.get('source', {}).get('title', 'Market News')

                # Clean title source suffix like "- Bloomberg"
                display_title = title
                if ' - ' in title:
                    parts = title.rsplit(' - ', 1)
                    display_title = parts[0]
                    if source == 'Market News' and len(parts) > 1:
                        source = parts[1]

                # Sentiment Scoring
                lower_title = title.lower()
                bull_count = sum(1 for kw in BULLISH_KEYWORDS_USD if kw.lower() in lower_title)
                bear_count = sum(1 for kw in BEARISH_KEYWORDS_USD if kw.lower() in lower_title)

                sentiment = 'NEUTRAL'
                score = 0.0
                if bull_count > bear_count:
                    sentiment = 'BULLISH (ドル高要因)'
                    score = min(1.0, 0.3 * (bull_count - bear_count))
                elif bear_count > bull_count:
                    sentiment = 'BEARISH (ドル安要因)'
                    score = max(-1.0, -0.3 * (bear_count - bull_count))

                news_items.append({
                    'title': display_title,
                    'link': link,
                    'published': published,
                    'source': source,
                    'sentiment': sentiment,
                    'score': score
                })

                if len(news_items) >= limit:
                    break
        except Exception:
            continue
        if len(news_items) >= limit:
            break

    # Calculate overall sentiment score
    total_score = sum(n['score'] for n in news_items)
    avg_score = total_score / len(news_items) if news_items else 0.0

    bullish_count = sum(1 for n in news_items if 'BULLISH' in n['sentiment'])
    bearish_count = sum(1 for n in news_items if 'BEARISH' in n['sentiment'])
    neutral_count = len(news_items) - bullish_count - bearish_count

    if avg_score > 0.15:
        overall_sentiment = 'BULLISH (ニュース優勢: ドル買い/円売り)'
    elif avg_score < -0.15:
        overall_sentiment = 'BEARISH (ニュース優勢: ドル売り/円買い)'
    else:
        overall_sentiment = 'NEUTRAL (ニュース拮抗 / 材料消化待ち)'

    summary = {
        'total_news': len(news_items),
        'avg_score': round(avg_score, 2),
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'neutral_count': neutral_count,
        'overall_sentiment': overall_sentiment
    }

    return news_items, summary
