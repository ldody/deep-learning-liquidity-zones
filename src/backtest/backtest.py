# import packages
import os, sys
import logging
import warnings
import pandas as pd
import numpy as np
import argparse
import pickle
import time
from joblib import Parallel, delayed

warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=FutureWarning)
		
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_log import Base, log_execution

class ohlcv_bid_ask():
	"""
	Clustering LOB and FO data.

	Args:
		job_id (int, optionnal): Slurm job ID.
	"""
	def __init__(self, job_id: int = 0, resampling_unit: str = 'min'):
		"""
		Initializing the regression instance.
		
		Attributes:

			
		Args:
			job_id (int, optionnal): Slurm job ID, Default=0.
		"""
		self.path = os.path.dirname(os.path.abspath(__file__))
		self.root_path = self.path
		while os.path.basename(self.root_path) != 'PhD_article_2':
			self.root_path =  os.path.dirname(self.root_path)
		self.job_id = job_id
		self.path_model = os.path.join(self.root_path,'model')
		self.data_path = os.path.join(self.root_path,'data')
		self.ohlcv_path = os.path.join(self.data_path,'raw','OHLCV')
		self.processed_path = os.path.join(self.data_path,'processed','FOB')
		self.processed_path_LOB = os.path.join(self.processed_path,'LOB')
		self.results_path = os.path.join(self.root_path,'results','comp_models')
		self.df_assets = pd.read_csv(os.path.join(self.data_path, 'assets.csv'), index_col=0)#[['ISIN','RIC']]
		self.files_input = pd.DataFrame(columns=['ISIN','data'])
		self.files = {}
		
		# --- Backtest hyperparameters ---
		self.horizon_steps = 50           # e.g. 3 * 5min; adapt if needed
		self.min_zone_width_ticks = 2    # discard zones narrower than 2 ticks
		self.widen_factor_spread = 0.5   # widen zones by 0.5 * spread on each side
		self.rng = np.random.default_rng(42)

	def get_files(self):
		"""
		Retrieving clustering files and OHLCV files.
		
		
		
		"""
		for path, f_type in zip([self.processed_path_LOB, self.ohlcv_path], ['LOB', 'OHLCV']):
			self.files[f_type] = [os.path.join(path, f) for f in os.listdir(path) if (f.split('_')[0] in self.df_assets.values.flatten().tolist())]
			
			if f_type == 'OHLCV':
				df_tmp = pd.DataFrame({f_type: self.files[f_type]})
				df_tmp['RIC'] = df_tmp[f_type].apply(lambda x: os.path.basename(x).split('_')[0])
				self.df_assets = self.df_assets.merge(df_tmp, on='RIC', how='left')
				
			else:
				df_tmp = pd.DataFrame({f_type: self.files[f_type]})
				df_tmp['ISIN'] = df_tmp[f_type].apply(lambda x: os.path.basename(x).split('_')[0])
				self.df_assets = self.df_assets.merge(df_tmp, on='ISIN', how='left')
				
		self.df_assets.set_index(['ISIN','RIC','OHLCV'], inplace=True)
		t = self.df_assets.loc[:,['LOB']]
		a = self.df_assets.loc[:,~self.df_assets.columns.isin(['LOB', 'FO'])].reset_index()
		t = t.stack().to_frame().reset_index()
		t.columns = ['ISIN', 'RIC', 'OHLCV', 'data_type', 'path']
		self.df_assets = t.merge(a, on=['ISIN','RIC','OHLCV'], how='left').reset_index()
		
	def load_data(self):
		"""
		Loading data.
		"""
		self.df_ohlcv = pd.read_csv(self.to_process['OHLCV'])
		self.df_ohlcv['Local Time'] = pd.to_datetime(self.df_ohlcv['Local Time'])
		self.df_ohlcv = self.df_ohlcv.set_index('Local Time').between_time('9:00', '17:00')
		#self.df_ohlcv = self.df_ohlcv['Volume' not in self.df_ohlcv.columns]
		
		self.df_data = pd.read_parquet(self.to_process['path'])

		self.df_data = self.df_data.between_time('9:00', '17:00').reset_index()
		
		def attrib(data):
			if 'Buy' in data['side'].values:
				return data.loc[data['price'] == data['price'].max()]
			
			elif 'Sell' in data['side'].values:
				return data.loc[data['price'] == data['price'].min()]
				
		self.df_data = self.df_data.reset_index().groupby(['index','side'], as_index=False).apply(attrib)
		self.df_data = self.df_data.reset_index(drop=True)
		self.df_data = self.df_data.groupby(['index','side']).last()
		self.df_data = self.df_data[['price','size']].unstack(level=1)
		self.df_data.columns = [f"{lvl0}_{lvl1}" for (lvl0, lvl1) in self.df_data.columns]
		self.df_data = self.df_data.rename(columns={
			"price_Buy": "bid",
			"price_Sell": "ask",
			"size_Buy": "bid_size",
			"size_Sell": "ask_size",
		})
				
		self.merged_df = self.df_ohlcv.join(self.df_data, how="inner").sort_index()
		self.merged_df["mid"] = (self.merged_df["bid"] + self.merged_df["ask"]) / 2.0
		
		y_pred = pd.read_csv(os.path.join(self.results_path, 'y_pred.csv'), index_col=0)
		y_pred.columns = ['lower_bound','upper_bound']
		
		dates = pd.read_csv(os.path.join(self.results_path, 'dates.csv'), index_col=0)
		asset = pd.read_csv(os.path.join(self.results_path, 'asset.csv'), index_col=0).map(lambda x: os.path.basename(os.path.normpath(x)).split('.')[0])
		asset.columns = ['asset']
		dates.columns = ['timestamp']
		dates['timestamp'] = pd.to_datetime(dates['timestamp'])

		dates['entity'] = (dates['timestamp'].diff() < pd.Timedelta(0)).fillna(True).cumsum()

		asset['entity'] = asset.index
		dates = dates.merge(asset[['entity', 'asset']], on='entity', how='left')

		dates = dates.loc[dates.index.repeat(10)].reset_index(drop=True)
		
		self.bounds = dates.join(y_pred, how='inner')
		self.bounds = self.bounds.drop(columns=['entity'], axis=1)
		
	# ------------------------------------------------------------------
	# ZONES FROM PREDICTIONS
	# ------------------------------------------------------------------
	def _build_zones_by_time_for_current_asset(self):
		"""
		Build a dict mapping each timestamp -> list of (low, high) zones
		for the asset currently in self.to_process.

		Uses:
			- self.merged_df  (has bid, ask, mid)
			- self.bounds     (has timestamp, asset, lower_bound, upper_bound)
			- self.to_process['Tick_step'] as tick size
			- self.min_zone_width_ticks, self.widen_factor_spread
		"""
		tick_size = float(self.to_process['Tick_step'])
		min_width = self.min_zone_width_ticks * tick_size

		ric = self.to_process['RIC']
		# adjust this line if your 'asset' in bounds is not the RIC root
		asset_name = ric.split('.')[0]

		bounds_asset = self.bounds[self.bounds['asset'] == asset_name].copy()
		if bounds_asset.empty:
			print(f"[WARN] No predicted bounds found for asset {asset_name}")
			self.zones_by_time = {}
			return

		zones_by_time = {}

		for t, grp in bounds_asset.groupby('timestamp'):
			t = pd.to_datetime(t)
			if t not in self.merged_df.index:
				continue

			spread = float(self.merged_df.loc[t, "ask"] - self.merged_df.loc[t, "bid"])
			zones = []

			for _, row in grp.iterrows():
				low = float(min(row["lower_bound"], row["upper_bound"]))
				high = float(max(row["lower_bound"], row["upper_bound"]))
				width = high - low

				# drop micro-zones narrower than min_width
				if width < min_width:
					continue

				# widen the zone using a fraction of the spread
				if spread > 0 and self.widen_factor_spread > 0:
					buffer_ = self.widen_factor_spread * spread
					low -= buffer_
					high += buffer_

				zones.append((low, high))

			if zones:
				zones_by_time[t] = zones

		self.zones_by_time = zones_by_time


	# ------------------------------------------------------------------
	# FILTER ZONES PER SIDE
	# ------------------------------------------------------------------
	def _filter_zones_for_side(self, zones, decision_mid, side: str):
		"""
		For BUY: keep zones at/above mid.
		For SELL: keep zones at/below mid.
		"""
		if side == "buy":
			return [(low, high) for (low, high) in zones if high >= decision_mid]
		elif side == "sell":
			return [(low, high) for (low, high) in zones if low <= decision_mid]
		else:
			raise ValueError("side must be 'buy' or 'sell'")

	# ------------------------------------------------------------------
	# BACKTEST ONE SIDE
	# ------------------------------------------------------------------
	def _run_backtest_one_side(self, side: str = "buy") -> pd.DataFrame:
		"""
		Run liquidity-aware execution backtest for a single side ("buy" or "sell")
		for the current asset.

		Returns
		-------
		DataFrame with columns:
			["decision_mid", "exec_immediate", "exec_random",
			 "exec_liq", "exec_vwap", "final_mid"]
		indexed by timestamp.
		"""
		if not hasattr(self, "zones_by_time"):
			self._build_zones_by_time_for_current_asset()

		rows = []
		common_times = sorted(set(self.merged_df.index) & set(self.zones_by_time.keys()))

		for t in common_times:
			pos = self.merged_df.index.get_loc(t)
			if pos + self.horizon_steps >= len(self.merged_df.index):
				continue

			decision_mid = float(self.merged_df.iloc[pos]["mid"])
			future = self.merged_df.iloc[pos + 1 : pos + 1 + self.horizon_steps].copy()
			future["mid"] = (future["bid"] + future["ask"]) / 2.0

			zones_all = self.zones_by_time.get(t, [])
			zones = self._filter_zones_for_side(zones_all, decision_mid, side)
			if not zones:
				# no relevant zones for this side at this time
				continue

			# ---------- Strategy 1: Immediate ----------
			exec_immediate = decision_mid

			# ---------- Strategy 2: Random ----------
			rand_idx = self.rng.integers(0, len(future))
			exec_random = float(future["mid"].iloc[rand_idx])

			# ---------- Strategy 3: Liquidity-aware ----------
			exec_liq = None
			for _, row in future.iterrows():
				m = float(row["mid"])
				hit = any((m >= low) and (m <= high) for (low, high) in zones)
				if hit:
					exec_liq = m
					break

			if exec_liq is None:
				exec_liq = float(future["mid"].iloc[-1])

			# ---------- Strategy 4: VWAP ----------
			# try to use volume column if present, else simple mean
			volume_col = None
			for cand in ["Volume", "volume", "VOL", "vol"]:
				if cand in future.columns:
					volume_col = cand
					break

			if volume_col is not None and future[volume_col].sum() > 0:
				exec_vwap = float((future["mid"] * future[volume_col]).sum() /
								  future[volume_col].sum())
			else:
				exec_vwap = float(future["mid"].mean())

			final_mid = float(future["mid"].iloc[-1])

			rows.append({
				"timestamp": t,
				"decision_mid": decision_mid,
				"exec_immediate": exec_immediate,
				"exec_random": exec_random,
				"exec_liq": exec_liq,
				"exec_vwap": exec_vwap,
				"final_mid": final_mid,
			})

		if not rows:
			return pd.DataFrame(columns=[
				"decision_mid", "exec_immediate", "exec_random",
				"exec_liq", "exec_vwap", "final_mid"
			])

		return pd.DataFrame(rows).set_index("timestamp")


	# ------------------------------------------------------------------
	# METRICS
	# ------------------------------------------------------------------
	def _compute_metrics(self, results: pd.DataFrame, side: str = "buy") -> pd.DataFrame:
		"""
		Add slippage & adverse selection columns to results for given side.

		For buy:
			slippage = (exec - decision) / decision
			adverse = final_mid < exec

		For sell:
			slippage = (decision - exec) / decision
			adverse = final_mid > exec
		"""
		out = results.copy()

		for col in ["exec_immediate", "exec_random", "exec_liq", "exec_vwap"]:
			if side == "buy":
				out[f"slip_{col}"] = (out[col] - out["decision_mid"]) / out["decision_mid"]
				out[f"adv_{col}"] = out["final_mid"] < out[col]
			else:  # sell
				out[f"slip_{col}"] = (out["decision_mid"] - out[col]) / out["decision_mid"]
				out[f"adv_{col}"] = out["final_mid"] > out[col]

		return out


	def _summarize(self, results: pd.DataFrame) -> pd.DataFrame:
		"""
		Summary table: mean slippage & adverse selection probability
		for all execution strategies.
		"""
		if results.empty:
			return pd.DataFrame(
				{"Mean slippage": [], "Adverse selection prob.": []},
				index=[]
			)

		return pd.DataFrame({
			"Mean slippage": [
				results["slip_exec_immediate"].mean(),
				results["slip_exec_random"].mean(),
				results["slip_exec_liq"].mean(),
				results["slip_exec_vwap"].mean(),
			],
			"Adverse selection prob.": [
				results["adv_exec_immediate"].mean(),
				results["adv_exec_random"].mean(),
				results["adv_exec_liq"].mean(),
				results["adv_exec_vwap"].mean(),
			],
		}, index=["Immediate", "Random", "Liquidity-aware", "VWAP"])


	# ------------------------------------------------------------------
	# PUBLIC: RUN BACKTEST FOR CURRENT ASSET
	# ------------------------------------------------------------------
	def run_backtest_current_asset(self):
		"""
		Run H4 backtest (buy & sell) for the current asset in self.to_process.
		Uses self.merged_df and self.bounds built in load_data().
		"""
		# Build zones
		self._build_zones_by_time_for_current_asset()

		# BUY side
		raw_buy = self._run_backtest_one_side(side="buy")
		metrics_buy = self._compute_metrics(raw_buy, side="buy")
		summary_buy = self._summarize(metrics_buy)

		# SELL side
		raw_sell = self._run_backtest_one_side(side="sell")
		metrics_sell = self._compute_metrics(raw_sell, side="sell")
		summary_sell = self._summarize(metrics_sell)

		print("\n=== BUY side summary ===")
		print(summary_buy)
		print("\n=== SELL side summary ===")
		print(summary_sell)

		# Optionally store on self for later saving
		self.metrics_buy = metrics_buy
		self.metrics_sell = metrics_sell
		self.summary_buy = summary_buy
		self.summary_sell = summary_sell

	
	def main(self):
		"""
		Launch H4 backtest for all assets and multiple horizons in parallel.
		"""
		self.get_files()

		HORIZONS = [10, 30, 50, 100, 150, 200, 250]   # 10, 30, 50 x 5min = robustness tests

		all_runs = []

		for H in HORIZONS:
			print(f"\n=== Running backtest for horizon_steps = {H} ===")

			summaries_list = Parallel(n_jobs=20)(
				delayed(run_for_asset)(row, H)
				for _, row in self.df_assets.iterrows()
			)

			# Concatenate across assets for this horizon
			summaries_H = pd.concat(summaries_list, ignore_index=True)
			all_runs.append(summaries_H)

		# Concatenate all horizons
		all_summaries = pd.concat(all_runs, ignore_index=True)

		# Save once
		out_path = os.path.join(self.results_path, "backtest_summaries_all_assets_robust.csv")
		all_summaries.to_csv(out_path, index=False)

		print("\n=== Global H4 robustness summary saved to ===")
		print(out_path)

			

def run_for_asset(asset_row, horizon_steps: int):
	"""
	Run the full H4 backtest for a single asset and a given horizon.

	Returns a tidy summary with columns:
	["asset", "side", "strategy", "horizon_steps",
	 "Mean slippage", "Adverse selection prob."]
	"""
	bt = ohlcv_bid_ask()
	bt.to_process = asset_row

	# set the horizon for this run
	bt.horizon_steps = horizon_steps

	bt.load_data()
	bt.run_backtest_current_asset()

	asset_ric = asset_row["RIC"]
	out = []

	# BUY side summary
	if hasattr(bt, "summary_buy") and bt.summary_buy is not None and not bt.summary_buy.empty:
		tmp = bt.summary_buy.copy()
		tmp["asset"] = asset_ric
		tmp["side"] = "buy"
		tmp["strategy"] = tmp.index
		tmp["horizon_steps"] = horizon_steps
		out.append(tmp.reset_index(drop=True))

	# SELL side summary
	if hasattr(bt, "summary_sell") and bt.summary_sell is not None and not bt.summary_sell.empty:
		tmp = bt.summary_sell.copy()
		tmp["asset"] = asset_ric
		tmp["side"] = "sell"
		tmp["strategy"] = tmp.index
		tmp["horizon_steps"] = horizon_steps
		out.append(tmp.reset_index(drop=True))

	if out:
		return pd.concat(out, ignore_index=True)
	else:
		return pd.DataFrame(
			columns=[
				"asset", "side", "strategy", "horizon_steps",
				"Mean slippage", "Adverse selection prob."
			]
		)



if __name__ == "__main__":
	
	ohlcv_bid_ask().main()