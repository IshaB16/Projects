import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# 1. FILE HANDLING – Load CSV Data

def load_transactions(filepath):
    """Read and validate the CSV file. Returns list of dicts."""
    transactions = []
    

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' not found. Please check the path.")

    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        required_columns = {'Date', 'Amount', 'Category'}
        if not required_columns.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must contain columns: {required_columns}")

        for i, row in enumerate(reader, start=2):  # start=2 (row 1 is header)
            try:
                date = datetime.strptime(row['Date'].strip(), '%Y-%m-%d')
                amount = float(row['Amount'].strip())
                category = row['Category'].strip().title()

                if amount <= 0:
                    print(f"  [Warning] Row {i}: Amount must be positive – skipping.")
                    continue

                transactions.append({
                    'date': date,
                    'amount': amount,
                    'category': category
                })

            except ValueError as e:
                print(f"  [Warning] Row {i}: Skipping invalid data – {e}")

    if not transactions:
        raise ValueError("No valid transactions found in the file.")

    return transactions

# 2. DATA PROCESSING – Categorization & Stats

def get_category_totals(transactions):
    """Return a dict of {category: total_amount}."""
    categories = {}
    for t in transactions:
        categories[t['category']] = categories.get(t['category'], 0) + t['amount']
    return categories


def get_daily_totals(transactions):
    """Return a dict of {date: total_amount}."""
    daily = {}
    for t in transactions:
        day = t['date'].date()
        daily[day] = daily.get(day, 0) + t['amount']
    return daily


def calculate_stats(transactions):
    """Calculate total, average, min, max spending."""
    amounts = list(map(lambda t: t['amount'], transactions))  # using map()
    total = sum(amounts)
    average = total / len(amounts)
    return {
        'total': total,
        'average': average,
        'min': min(amounts),
        'max': max(amounts),
        'count': len(amounts)
    }


# 3. ANOMALY DETECTION (Main Feature)

def detect_anomalies(transactions, multiplier=2.0):
    """
    Flag a transaction as anomaly if its amount >
    (average + multiplier * std_deviation).

    Uses filter() for functional programming concept.
    """
    amounts = list(map(lambda t: t['amount'], transactions))
    avg = sum(amounts) / len(amounts)

    # Calculate standard deviation manually (no numpy required)
    variance = sum((x - avg) ** 2 for x in amounts) / len(amounts)
    std_dev = variance ** 0.5

    threshold = avg + multiplier * std_dev

    # Using filter() – functional programming concept
    anomalies = list(filter(lambda t: t['amount'] > threshold, transactions))

    return anomalies, threshold, avg, std_dev


# 4. DATA VISUALIZATION – Matplotlib Graphs

def plot_all(transactions, anomalies, category_totals, daily_totals, output_dir='.'):
    """Generate and save 3 charts: bar, line, and anomaly scatter."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Smart Expense Anomaly Detector – Spending Insights',
                 fontsize=14, fontweight='bold', y=1.02)

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52',
              '#8172B3', '#937860', '#DA8BC3', '#8C8C8C']

    # ── Chart 1: Category-wise Bar Chart ──
    ax1 = axes[0]
    cats = list(category_totals.keys())
    vals = list(category_totals.values())
    bar_colors = colors[:len(cats)]
    bars = ax1.bar(cats, vals, color=bar_colors, edgecolor='white', linewidth=0.8)
    ax1.set_title('Category-wise Spending', fontweight='bold')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Amount (₹)')
    ax1.tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(vals) * 0.01,
                 f'₹{val:,.0f}', ha='center', va='bottom', fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── Chart 2: Daily Spending Line Chart ──
    ax2 = axes[1]
    sorted_days = sorted(daily_totals.keys())
    day_amounts = [daily_totals[d] for d in sorted_days]
    ax2.plot(sorted_days, day_amounts, marker='o', color='#4C72B0',
             linewidth=2, markersize=5, markerfacecolor='white', markeredgewidth=1.5)
    ax2.fill_between(sorted_days, day_amounts, alpha=0.12, color='#4C72B0')
    ax2.set_title('Daily Spending Trend', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Amount (₹)')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(sorted_days) // 7)))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ── Chart 3: Anomaly Detection Scatter ──
    ax3 = axes[2]
    dates = [t['date'].date() for t in transactions]
    amounts = [t['amount'] for t in transactions]
    anomaly_dates = [t['date'].date() for t in anomalies]
    anomaly_amounts = [t['amount'] for t in anomalies]

    avg = sum(amounts) / len(amounts)
    _, threshold, _, _ = detect_anomalies(transactions)

    ax3.scatter(dates, amounts, color='#4C72B0', s=60, zorder=3, label='Normal', alpha=0.85)
    if anomaly_dates:
        ax3.scatter(anomaly_dates, anomaly_amounts,
                    color='#C44E52', s=100, zorder=4,
                    marker='^', label='Anomaly', edgecolors='darkred', linewidth=0.8)
    ax3.axhline(y=threshold, color='#C44E52', linestyle='--',
                linewidth=1.2, label=f'Threshold ₹{threshold:,.0f}')
    ax3.axhline(y=avg, color='#55A868', linestyle='--',
                linewidth=1.2, label=f'Average ₹{avg:,.0f}')
    ax3.set_title('Anomaly Detection', fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Amount (₹)')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(sorted_days) // 7)))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax3.legend(fontsize=8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'expense_insights.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n  Charts saved → {chart_path}")
    plt.show()


# 5. REPORT OUTPUT – Console Summary

def print_report(transactions, stats, category_totals, anomalies, threshold):
    """Print a clean summary report to the console."""

    divider = '─' * 50

    print('\n' + '═' * 50)
    print('   SMART EXPENSE ANOMALY DETECTOR – REPORT')
    print('═' * 50)

    # Overall stats
    print(f'\n OVERALL SUMMARY')
    print(divider)
    print(f"  Total Transactions : {stats['count']}")
    print(f"  Total Spending     : ₹{stats['total']:,.2f}")
    print(f"  Average per Txn    : ₹{stats['average']:,.2f}")
    print(f"  Lowest Transaction : ₹{stats['min']:,.2f}")
    print(f"  Highest Transaction: ₹{stats['max']:,.2f}")

    # Category breakdown
    print(f'\n🗂  CATEGORY-WISE BREAKDOWN')
    print(divider)
    sorted_cats = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
    for cat, total in sorted_cats:
        pct = (total / stats['total']) * 100
        bar = '█' * int(pct / 5)
        print(f"  {cat:<15} ₹{total:>10,.2f}  ({pct:5.1f}%)  {bar}")
    top_cat = sorted_cats[0][0]
    print(f"\n    Highest spending category: {top_cat}")

    # Anomalies
    print(f'\n ANOMALY DETECTION')
    print(divider)
    print(f"  Detection threshold: ₹{threshold:,.2f}")
    print(f"  Anomalies found    : {len(anomalies)}")

    if anomalies:
        print()
        for a in sorted(anomalies, key=lambda x: x['amount'], reverse=True):
            print(f"  ⚡ ₹{a['amount']:>10,.2f}  |  {a['date'].strftime('%Y-%m-%d')}  |  {a['category']}")
    else:
        print("  ✅ No unusual spending detected.")

    print('\n' + '═' * 50)
    print("  Graphs saved as  expense_insights.png")
    print('═' * 50 + '\n')


# 6. MAIN – Entry Point with Exception Handling

def main():
    print('\n' + '═' * 50)
    print('   Smart Expense Anomaly Detector')
    print('   Starting analysis...')
    print('═' * 50)

    # ── Get CSV path from user ──
    filepath = input("\n  Enter path to your CSV file (or press Enter for demo): ").strip()

    # ── If no file given, create a demo CSV ──
    if not filepath:
        filepath = 'demo_expenses.csv'
        demo_data = """Date,Amount,Category
2026-03-01,200,Food
2026-03-02,1500,Shopping
2026-03-03,100,Travel
2026-03-04,5000,Shopping
2026-03-05,300,Food
2026-03-06,450,Bills
2026-03-07,150,Travel
2026-03-08,800,Shopping
2026-03-09,250,Food
2026-03-10,4800,Bills
2026-03-11,200,Food
2026-03-12,350,Travel
2026-03-13,120,Food
2026-03-14,600,Shopping
2026-03-15,180,Travel
"""
        with open(filepath, 'w') as f:
            f.write(demo_data)
        print(f"\n  Demo file created: {filepath}")

    # ── Run pipeline with exception handling ──
    try:
        print("\n  Loading transactions...")
        transactions = load_transactions(filepath)
        print(f"   {len(transactions)} valid transactions loaded.")

        print("  Processing data...")
        stats = calculate_stats(transactions)
        category_totals = get_category_totals(transactions)
        daily_totals = get_daily_totals(transactions)

        print("  Detecting anomalies...")
        anomalies, threshold, avg, std_dev = detect_anomalies(transactions, multiplier=2.0)

        print("  Generating report...")
        print_report(transactions, stats, category_totals, anomalies, threshold)

        print("  Generating charts...")
        plot_all(transactions, anomalies, category_totals, daily_totals)

    except FileNotFoundError as e:
        print(f"\n   File Error: {e}")

    except ValueError as e:
        print(f"\n   Data Error: {e}")

    except Exception as e:
        print(f"\n   Unexpected error: {e}")


if __name__ == '__main__':
    main()