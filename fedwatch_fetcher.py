"""
CME FedWatch data fetcher and monetary policy sentiment analysis.
Extracts next FOMC probabilities and calculates market policy bias on USD.
"""
from datetime import datetime, date


def fetch_fedwatch_analysis() -> dict:
    """
    Fetches latest CME FedWatch data and interprets monetary policy bias.
    """
    result = {
        'status': 'success',
        'current_effr': 3.88,
        'current_target': '3.75% - 4.00%',
        'next_meeting_date': None,
        'days_to_meeting': None,
        'probabilities': {},
        'hold_prob': 0.0,
        'cut_prob': 0.0,
        'hike_prob': 0.0,
        'stance': 'Neutral (中立)',
        'us_bias_score': 0.0,  # +1.0 = Strong USD bias (Hawkish), -1.0 = Weak USD bias (Dovish)
        'summary_text': '',
        'all_meetings': []
    }

    try:
        import cme_fedwatch as cfw
        probs_data = cfw.get_probabilities()

        if probs_data and 'meetings' in probs_data:
            result['current_effr'] = probs_data.get('effr', 3.88)
            result['current_target'] = probs_data.get('current_target', '3.75%-4.00%')

            meetings = probs_data['meetings']
            if meetings:
                next_m = meetings[0]
                m_date_str = next_m['date']
                result['next_meeting_date'] = m_date_str

                # Calculate days to next meeting
                try:
                    m_date = datetime.strptime(m_date_str, '%Y-%m-%d').date()
                    today = date.today()
                    result['days_to_meeting'] = (m_date - today).days
                except Exception:
                    result['days_to_meeting'] = 30

                probs = next_m.get('probabilities', {})
                result['probabilities'] = probs

                # Current target range parsing, e.g. "3.75%-4.00%"
                curr_target = result['current_target'].replace(' ', '')
                # Parse lower bound
                try:
                    curr_lower = float(curr_target.split('%')[0])
                except Exception:
                    curr_lower = 3.75

                hold_p = 0.0
                cut_p = 0.0
                hike_p = 0.0

                for rate_range, p in probs.items():
                    try:
                        range_clean = rate_range.replace('%', '').strip()
                        parts = range_clean.split('-')
                        range_lower = float(parts[0])

                        if abs(range_lower - curr_lower) < 0.1:
                            hold_p += p
                        elif range_lower < curr_lower:
                            cut_p += p
                        else:
                            hike_p += p
                    except Exception:
                        pass

                result['hold_prob'] = round(hold_p, 1)
                result['cut_prob'] = round(cut_p, 1)
                result['hike_prob'] = round(hike_p, 1)

                # Store top 4 upcoming meetings for display
                result['all_meetings'] = meetings[:4]

                # Determine stance & USD bias score
                # Hike > 50% -> Hawkish (+0.8)
                # Cut > 50% -> Dovish (-0.8)
                # Hold > 50% -> Neutral to slightly Hawkish/Dovish based on balance
                bias = (hike_p - cut_p) / 100.0
                # Clamp between -1.0 and 1.0
                result['us_bias_score'] = max(-1.0, min(1.0, bias))

                if hike_p >= 50:
                    result['stance'] = 'Hawkish / 利上げ観測優勢 (ドル高圧力)'
                    result['summary_text'] = f"次回FOMC（{m_date_str}）において{hike_p:.1f}%の確率で利上げが織り込まれており、金利差拡大から米ドル買い（USD/JPY上昇）圧力が強く働きやすい状態です。"
                elif cut_p >= 50:
                    result['stance'] = 'Dovish / 利下げ観測優勢 (ドル安圧力)'
                    result['summary_text'] = f"次回FOMC（{m_date_str}）において{cut_p:.1f}%の確率で利下げが予想されており、金利低下観測から米ドル売り（USD/JPY下落）が意識されます。"
                elif hike_p > cut_p:
                    result['stance'] = 'Neutral-Hawkish / 据え置き〜引き締め維持'
                    result['summary_text'] = f"次回FOMC（{m_date_str}）は据え置き（{hold_p:.1f}%）が中心ですが、引き締め警戒（{hike_p:.1f}%）が残り、ドルの底堅さをサポートしています。"
                elif cut_p > hike_p:
                    result['stance'] = 'Neutral-Dovish / 据え置き〜利下げバイアス'
                    result['summary_text'] = f"次回FOMC（{m_date_str}）は据え置き（{hold_p:.1f}%）が本命ながら利下げ観測（{cut_p:.1f}%）が意識され、上値の重しとなりやすい環境です。"
                else:
                    result['stance'] = 'Neutral / 金利据え置き予想'
                    result['summary_text'] = f"次回FOMC（{m_date_str}）は金利据え置き予想が{hold_p:.1f}%で圧倒的であり、金利観測による短期的インパクトは中立的です。"

    except Exception as e:
        result['status'] = 'fallback'
        result['stance'] = 'Neutral (観測安定)'
        result['summary_text'] = f"CME FedWatchデータ接続中（デフォルト推定値適用）: {e}"
        result['hold_prob'] = 85.0
        result['cut_prob'] = 15.0
        result['us_bias_score'] = -0.15

    return result
