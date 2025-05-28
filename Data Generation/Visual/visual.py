import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import sys
import os

# AI: Add parent directory to path to import generate1
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate1 import generate_deterministic_stock_data

# AI: Type definitions for clarity
DayNumber = int
PriceValue = float
EventValue = float

def create_summary_analysis(stock_df: pd.DataFrame, events_log: List[Dict[str, Any]], graphs_dir: str) -> None:
    """
    Creates summary analysis showing why this data is challenging for models
    """
    
    # AI: Calculate statistics that demonstrate complexity
    event_df = pd.DataFrame(events_log)
    
    # AI: Count sequence-dependent events
    sequence_dependent_events = len([e for e in events_log if 'sequence_factor' in e])
    total_events = len(events_log)
    
    # AI: Calculate impact variance by event type
    impact_variance_by_type = event_df.groupby('type')['impact'].var()
    
    # AI: Calculate correlation between event values and impacts (should be weak due to sequence effects)
    correlation_value_impact = event_df['value'].corr(event_df['impact'])
    
    # AI: Create summary figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # AI: 1. Event type distribution
    event_counts = event_df['type'].value_counts()
    ax1.pie(event_counts.values, labels=event_counts.index, autopct='%1.1f%%')
    ax1.set_title('Event Type Distribution')
    
    # AI: 2. Sequence dependency frequency
    sequence_counts = [sequence_dependent_events, total_events - sequence_dependent_events]
    ax2.pie(sequence_counts, labels=['Sequence Dependent', 'Independent'], 
            autopct='%1.1f%%', colors=['red', 'blue'])
    ax2.set_title('Sequence Dependency\n({}/{} events affected)'.format(sequence_dependent_events, total_events))
    
    # AI: 3. Impact variance by event type
    impact_variance_by_type.plot(kind='bar', ax=ax3)
    ax3.set_title('Impact Variance by Event Type\n(High variance = sequence effects)')
    ax3.set_ylabel('Impact Variance')
    ax3.tick_params(axis='x', rotation=45)
    
    # AI: 4. Value-Impact correlation weakness
    ax4.scatter(event_df['value'], event_df['impact'], alpha=0.6)
    ax4.set_xlabel('Event Value')
    ax4.set_ylabel('Event Impact')
    ax4.set_title('Value-Impact Correlation: {:.3f}\n(Weak due to sequence effects)'.format(correlation_value_impact))
    
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'summary_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # AI: Print analysis summary
    print("\n" + "="*80)
    print("COMPLEXITY ANALYSIS: Why this data is challenging for simple models")
    print("="*80)
    print("Total events: {}".format(total_events))
    print("Sequence-dependent events: {} ({:.1f}%)".format(sequence_dependent_events, sequence_dependent_events/total_events*100))
    print("Value-Impact correlation: {:.3f} (weak due to sequence effects)".format(correlation_value_impact))
    
    print("\nImpact variance by event type:")
    for event_type, variance in impact_variance_by_type.items():
        print("  {}: {:.3f}".format(event_type, variance))
    
    print("\nWhy simple visualization fails:")
    print("- Price movements appear random without event context")
    print("- Event timing alone doesn't explain impact magnitude")
    print("- Same event types have different impacts based on history")
    
    print("\nWhy sequence-only models fail:")
    print("- Event values (board vote %, anticipation score, etc.) matter")
    print("- Same sequence of event types can have different outcomes")
    print("- Cross-event dependencies (CEO changes affect competitor vulnerability)")
    
    print("\nWhy value-only models fail:")
    print("- Same event values can have different impacts based on sequence")
    print("- Historical context changes the meaning of current events")
    print("- Temporal dependencies span multiple event cycles")

def create_comprehensive_visualizations():
    """
    Creates visualizations that demonstrate why this data is challenging for:
    1. Simple value-only visualization (missing sequence dependencies)
    2. Sequence-only models (missing event values and their correlations)
    """
    
    # AI: Generate the data
    stock_df, events_log = generate_deterministic_stock_data(num_days=200)
    
    # AI: Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # AI: Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    
    # AI: 1. SIMPLE PLOT - Shows why value-only visualization fails
    ax1 = plt.subplot(4, 2, 1)
    plt.plot(stock_df.index, stock_df['Price'], linewidth=2, alpha=0.8)
    plt.title('1. SIMPLE PRICE PLOT\n(Fails to show event causality)', fontsize=14, fontweight='bold')
    plt.xlabel('Day')
    plt.ylabel('Stock Price')
    plt.grid(True, alpha=0.3)
    
    # AI: 2. EVENTS OVERLAY - Shows timing but not values or dependencies
    ax2 = plt.subplot(4, 2, 2)
    plt.plot(stock_df.index, stock_df['Price'], linewidth=2, alpha=0.8, label='Price')
    
    # AI: Mark events by type with different colors
    event_colors = {'CEO_Change': 'red', 'Product_Launch': 'green', 
                   'Earnings_Report': 'blue', 'Competitor_Action': 'orange'}
    
    for event in events_log:
        day = event['day']
        event_type = event['type']
        plt.axvline(x=day, color=event_colors[event_type], alpha=0.6, linestyle='--')
    
    plt.title('2. EVENTS TIMING OVERLAY\n(Shows when but not why)', fontsize=14, fontweight='bold')
    plt.xlabel('Day')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # AI: 3. EVENT VALUES vs IMPACT - Shows correlation but not sequence dependencies
    ax3 = plt.subplot(4, 2, 3)
    event_df = pd.DataFrame(events_log)
    
    for event_type in event_colors.keys():
        type_events = event_df[event_df['type'] == event_type]
        if not type_events.empty:
            plt.scatter(type_events['value'], type_events['impact'], 
                       label=event_type, alpha=0.7, s=60)
    
    plt.title('3. EVENT VALUES vs IMPACT\n(Missing sequence context)', fontsize=14, fontweight='bold')
    plt.xlabel('Event Value')
    plt.ylabel('Impact on Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # AI: 4. SEQUENCE DEPENDENCY VISUALIZATION - Shows why sequence matters
    ax4 = plt.subplot(4, 2, 4)
    
    # AI: Extract CEO changes and their sequence factors
    ceo_events = [e for e in events_log if e['type'] == 'CEO_Change']
    ceo_days = [e['day'] for e in ceo_events]
    ceo_impacts = [e['impact'] for e in ceo_events]
    ceo_factors = [e.get('sequence_factor', 'unknown') for e in ceo_events]
    
    # AI: Color by sequence factor
    factor_colors = {'recent_CEO_change_instability': 'red', 'stable_CEO_change': 'blue'}
    for i, (day, impact, factor) in enumerate(zip(ceo_days, ceo_impacts, ceo_factors)):
        color = factor_colors.get(factor, 'gray')
        plt.scatter(day, impact, color=color, s=100, alpha=0.8)
        plt.annotate(factor[:10] + '...', (day, impact), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    plt.title('4. CEO CHANGE SEQUENCE DEPENDENCIES\n(Same event type, different impacts)', fontsize=14, fontweight='bold')
    plt.xlabel('Day')
    plt.ylabel('CEO Change Impact')
    plt.grid(True, alpha=0.3)
    
    # AI: 5. PRODUCT LAUNCH SUCCESS TRACKING
    ax5 = plt.subplot(4, 2, 5)
    
    product_events = [e for e in events_log if e['type'] == 'Product_Launch']
    product_days = [e['day'] for e in product_events]
    product_impacts = [e['impact'] for e in product_events]
    product_factors = [e.get('sequence_factor', 'baseline') for e in product_events]
    
    # AI: Show how past success affects current impact
    factor_colors_product = {
        'recent_product_success_boost': 'green',
        'recent_product_failure_penalty': 'red',
        'baseline': 'blue'
    }
    
    for day, impact, factor in zip(product_days, product_impacts, product_factors):
        color = factor_colors_product.get(factor, 'gray')
        plt.scatter(day, impact, color=color, s=100, alpha=0.8)
    
    plt.title('5. PRODUCT LAUNCH SEQUENCE EFFECTS\n(Past success affects current impact)', fontsize=14, fontweight='bold')
    plt.xlabel('Day')
    plt.ylabel('Product Launch Impact')
    plt.grid(True, alpha=0.3)
    
    # AI: 6. EARNINGS MOMENTUM VISUALIZATION
    ax6 = plt.subplot(4, 2, 6)
    
    earnings_events = [e for e in events_log if e['type'] == 'Earnings_Report']
    earnings_days = [e['day'] for e in earnings_events]
    earnings_values = [e['value'] for e in earnings_events]
    earnings_impacts = [e['impact'] for e in earnings_events]
    
    # AI: Show earnings value vs impact with momentum effects
    momentum_events = [e for e in earnings_events if 'sequence_factor' in e]
    regular_events = [e for e in earnings_events if 'sequence_factor' not in e]
    
    if regular_events:
        reg_values = [e['value'] for e in regular_events]
        reg_impacts = [e['impact'] for e in regular_events]
        plt.scatter(reg_values, reg_impacts, color='blue', alpha=0.6, s=60, label='Regular')
    
    if momentum_events:
        mom_values = [e['value'] for e in momentum_events]
        mom_impacts = [e['impact'] for e in momentum_events]
        plt.scatter(mom_values, mom_impacts, color='red', alpha=0.8, s=100, label='Momentum Effect')
    
    plt.title('6. EARNINGS MOMENTUM EFFECTS\n(Same values, different impacts)', fontsize=14, fontweight='bold')
    plt.xlabel('Earnings Surprise Value')
    plt.ylabel('Impact on Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # AI: 7. VULNERABILITY STATE ANALYSIS
    ax7 = plt.subplot(4, 2, 7)
    
    competitor_events = [e for e in events_log if e['type'] == 'Competitor_Action']
    comp_days = [e['day'] for e in competitor_events]
    comp_values = [e['value'] for e in competitor_events]
    comp_impacts = [e['impact'] for e in competitor_events]
    comp_factors = [e.get('sequence_factor', 'normal') for e in competitor_events]
    
    # AI: Show how company vulnerability affects competitor impact
    vulnerable_events = [e for e in competitor_events if 'vulnerable' in e.get('sequence_factor', '')]
    normal_events = [e for e in competitor_events if 'vulnerable' not in e.get('sequence_factor', '')]
    
    if normal_events:
        norm_values = [e['value'] for e in normal_events]
        norm_impacts = [e['impact'] for e in normal_events]
        plt.scatter(norm_values, norm_impacts, color='blue', alpha=0.6, s=60, label='Normal State')
    
    if vulnerable_events:
        vuln_values = [e['value'] for e in vulnerable_events]
        vuln_impacts = [e['impact'] for e in vulnerable_events]
        plt.scatter(vuln_values, vuln_impacts, color='red', alpha=0.8, s=100, label='Vulnerable State')
    
    plt.title('7. VULNERABILITY STATE EFFECTS\n(Company state affects competitor impact)', fontsize=14, fontweight='bold')
    plt.xlabel('Competitor Action Severity')
    plt.ylabel('Impact on Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # AI: 8. COMPREHENSIVE TIMELINE WITH ALL FACTORS
    ax8 = plt.subplot(4, 2, 8)
    
    # AI: Plot price with all events and their sequence factors
    plt.plot(stock_df.index, stock_df['Price'], linewidth=2, alpha=0.8, color='black', label='Price')
    
    # AI: Add event markers with size proportional to impact and color by sequence factor
    for event in events_log:
        day = event['day']
        impact = abs(event['impact'])
        event_type = event['type']
        has_sequence_factor = 'sequence_factor' in event
        
        marker_size = min(200, max(20, impact * 30))
        marker_color = 'red' if has_sequence_factor else event_colors[event_type]
        marker_alpha = 0.8 if has_sequence_factor else 0.4
        
        plt.scatter(day, stock_df.loc[day, 'Price'], s=marker_size, 
                   color=marker_color, alpha=marker_alpha, edgecolors='black', linewidth=0.5)
    
    plt.title('8. COMPLETE TIMELINE\n(Red = sequence-dependent events)', fontsize=14, fontweight='bold')
    plt.xlabel('Day')
    plt.ylabel('Stock Price')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # AI: Save to graphs folder
    graphs_dir = os.path.join(os.path.dirname(__file__), 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    
    plt.savefig(os.path.join(graphs_dir, 'comprehensive_stock_analysis.png'), 
                dpi=300, bbox_inches='tight')
    
    # AI: Create summary statistics
    create_summary_analysis(stock_df, events_log, graphs_dir)

if __name__ == "__main__":
    create_comprehensive_visualizations() 