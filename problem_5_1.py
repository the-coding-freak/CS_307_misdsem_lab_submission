import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
import warnings
import os
import sys
from contextlib import redirect_stderr

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class MarketRegimeAnalyzer:
    """Performs a comprehensive market regime analysis using Gaussian HMM."""

    def __init__(self, ticker, start_date="2010-01-01"):
        self.ticker = ticker
        self.start_date = start_date
        self.prices, self.returns = None, None
        self.models, self.model_scores = {}, {}
        self.best_model, self.best_n_states = None, None
        self.hidden_states, self.regime_names = None, None

    def load_data(self):
        print("\n" + "=" * 60 + "\nPART 1: DATA ACQUISITION & PREPROCESSING\n" + "=" * 60)
        print(f"Downloading data for {self.ticker}...")
        data = yf.download(self.ticker, start=self.start_date, progress=False, auto_adjust=True)
        if data.empty: raise ValueError(f"No data for {self.ticker}.")
        self.prices = data['Close'].copy()
        self.returns = self.prices.pct_change().dropna()
        print(f"Data loaded: {len(self.prices)} points from {self.prices.index[0].date()} to {self.prices.index[-1].date()}")

    def _fit_hmm_with_restarts(self, n_states, n_restarts=10):
        X = self.returns.values.reshape(-1, 1)
        best_model, best_score = None, -np.inf
        for i in range(n_restarts):
            try:
                model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=RANDOM_SEED + i)
                with open(os.devnull, 'w') as f, redirect_stderr(f):
                    model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score, best_model = score, model
            except Exception:
                continue
        return best_model, best_score

    def model_selection(self, min_states=2, max_states=5):
        print("\n" + "=" * 60 + "\nPART 2: MODEL SELECTION (2-5 STATES)\n" + "=" * 60)
        X, n_samples = self.returns.values.reshape(-1, 1), len(self.returns)
        results = []
        for n_states in range(min_states, max_states + 1):
            model, log_likelihood = self._fit_hmm_with_restarts(n_states)
            if model is None: continue
            n_params = (n_states ** 2 + 2 * n_states - 2)
            bic = np.log(n_samples) * n_params - 2 * log_likelihood
            self.models[n_states] = model
            results.append({'n_states': n_states, 'BIC': bic})
        
        results_df = pd.DataFrame(results)
        if results_df.empty: raise RuntimeError("All HMM model fittings failed.")
        
        best_row = results_df.loc[results_df['BIC'].idxmin()]
        self.best_n_states = int(best_row['n_states'])
        self.best_model = self.models[self.best_n_states]
        print("\nModel Selection Summary (lower BIC is better):\n" + results_df.to_string(index=False))
        print(f"\nBest model selected: {self.best_n_states} states.")

    def analyze_and_plot_model(self, model, n_states, suffix=""):
        """Analyzes a specific model and generates its plots."""
        print(f"\n" + "=" * 60 + f"\nANALYSIS FOR {n_states}-STATE MODEL\n" + "=" * 60)
        X = self.returns.values.reshape(-1, 1)
        
        try: vars_list = np.array([c[0][0] for c in model.covars_])
        except: vars_list = np.array([c[0] for c in model.covars_])
        sorted_idx = np.argsort(vars_list)
        
        state_mapping = {old: new for new, old in enumerate(sorted_idx)}
        hidden_states = np.array([state_mapping[s] for s in model.predict(X)])
        
        regime_names = [f"Regime {i+1}" for i in range(n_states)]
        if n_states > 0: regime_names[0] = "Low Volatility"
        if n_states > 1: regime_names[-1] = "High Volatility"

        # Generate Price vs. Regime Plot
        self._generate_price_plot(hidden_states, regime_names, n_states, suffix)
        
        # Generate Returns Distribution Plot
        self._generate_distribution_plot(hidden_states, regime_names, n_states, suffix)

    def _generate_price_plot(self, hidden_states, regime_names, n_states, suffix):
        safe_ticker = self.ticker.replace('^', '').replace('/', '_')
        colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, n_states))
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(self.prices.index, self.prices.values, 'k-', linewidth=1)
        y_min, y_max = self.prices.min(), self.prices.max()
        for i in range(n_states):
            ax.fill_between(self.returns.index, y_min, y_max, where=hidden_states==i, color=colors[i], alpha=0.4)
        
        patches = [mpatches.Patch(color=colors[i], alpha=0.4, label=regime_names[i]) for i in range(n_states)]
        ax.legend(handles=patches, loc='upper left')
        ax.set_title(f'{self.ticker} Price with {n_states}-State Regimes'); ax.set_ylabel('Price')
        filename = f"{safe_ticker}_price{suffix}.png"
        plt.savefig(filename); plt.show()
        print(f"OK Generated: {filename}")

    def _generate_distribution_plot(self, hidden_states, regime_names, n_states, suffix):
        safe_ticker = self.ticker.replace('^', '').replace('/', '_')
        colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, n_states))
        fig, ax = plt.subplots(figsize=(12, 7))
        for i in range(n_states):
            mask = hidden_states == i
            sns.kdeplot(self.returns[mask], ax=ax, label=regime_names[i], color=colors[i], fill=True, alpha=0.5)
        
        ax.set_title(f'Returns Distribution by Regime ({n_states}-State Model)'); ax.set_xlabel('Daily Returns'); ax.set_ylabel('Density')
        ax.legend()
        filename = f"{safe_ticker}_returns_dist{suffix}.png"
        plt.savefig(filename); plt.show()
        print(f"OK Generated: {filename}")


def run_comparative_analysis(tickers, start_date):
    """Bonus Task: Compare HMM results across different instruments."""
    print("\n" + "=" * 80 + "\nBONUS: COMPARATIVE ANALYSIS ACROSS DIVERSE INSTRUMENTS\n" + "=" * 80)
    comparison_results = []
    for ticker in tickers:
        try:
            print(f"\n--- Analyzing {ticker} for comparison ---")
            analyzer = MarketRegimeAnalyzer(ticker, start_date)
            analyzer.load_data()
            analyzer.model_selection(min_states=2, max_states=4)
            
            # We need to predict states for the best model to get the frequency
            X = analyzer.returns.values.reshape(-1, 1)
            hidden_states = analyzer.best_model.predict(X)
            high_vol_freq = 100 * np.mean(hidden_states == analyzer.best_n_states - 1)
            comparison_results.append({'Ticker': ticker, 'Best # States': analyzer.best_n_states, 'High Vol Freq.': high_vol_freq})
        except Exception as e:
            print(f"Could not analyze {ticker}: {e}")

    if not comparison_results: return
    df = pd.DataFrame(comparison_results)
    print("\nComparative Summary:\n" + df.to_string(index=False))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df['Ticker'], df['High Vol Freq.'], color='coral')
    ax.set_xlabel('Ticker'); ax.set_ylabel('Frequency of High Volatility (%)')
    ax.set_title('Comparative Analysis: High Volatility Regime Frequency')
    filename = "bonus_comparative_summary.png"
    plt.savefig(filename); plt.show()
    print(f"\nOK Generated: {filename}")


# Main execution
if __name__ == "__main__":
    
    # === PART 1: DETAILED ANALYSIS FOR PRIMARY TICKERS ===
    tickers_to_analyze = ["^GSPC", "TSLA", "AAPL"]
    start_date = "2010-01-01"
    
    for ticker in tickers_to_analyze:
        print(f"\n{'='*80}\nSTARTING DETAILED ANALYSIS FOR: {ticker}\n{'='*80}")
        try:
            analyzer = MarketRegimeAnalyzer(ticker, start_date)
            analyzer.load_data()
            analyzer.model_selection(min_states=2, max_states=5)

            # Generate specific plots for 2 and 3 state models as requested
            if 2 in analyzer.models:
                analyzer.analyze_and_plot_model(analyzer.models[2], 2, suffix="_2_state_model")
            if 3 in analyzer.models:
                analyzer.analyze_and_plot_model(analyzer.models[3], 3, suffix="_3_state_model")

        except Exception as e:
            print(f"\nAN ERROR OCCURRED during analysis for {ticker}: {e}")

    # === PART 2: BROADER COMPARATIVE ANALYSIS ON DIFFERENT INSTRUMENTS ===
    tickers_for_comparison = ["^IXIC", "GC=F", "JPM"]  # NASDAQ, Gold, JPMorgan Chase
    run_comparative_analysis(tickers_for_comparison, start_date)