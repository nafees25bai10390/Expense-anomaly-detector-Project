"""
src/model.py
Anomaly detection pipeline combining K-Means clustering and Isolation Forest.

Approach
--------
1. K-Means  — assigns each transaction to a cluster and computes its
   distance from the cluster centroid.  Transactions far from any
   centroid are flagged as potential anomalies.

2. Isolation Forest — an ensemble method that explicitly models anomalies
   as points that are easy to isolate in random decision trees.

3. Ensemble vote — a transaction is marked anomalous when BOTH methods
   agree, reducing false positives.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix


# ── Hyper-parameters (tune these if you want to experiment) ───────────────
KMEANS_N_CLUSTERS   = 7       # one cluster roughly per spending category
KMEANS_THRESHOLD_Q  = 0.92    # flag the top N% by centroid distance
IFOREST_CONTAMINATION = 0.05  # expected fraction of anomalies in data
RANDOM_SEED         = 42


class ExpenseAnomalyDetector:
    """
    Detects anomalous personal expense transactions using an ensemble of
    K-Means clustering and Isolation Forest.

    Attributes
    ----------
    kmeans       : fitted KMeans model
    iforest      : fitted IsolationForest model
    km_threshold : distance cut-off for the K-Means flag
    """

    def __init__(
        self,
        n_clusters:    int   = KMEANS_N_CLUSTERS,
        km_threshold_q: float = KMEANS_THRESHOLD_Q,
        if_contamination: float = IFOREST_CONTAMINATION,
    ):
        self.n_clusters        = n_clusters
        self.km_threshold_q    = km_threshold_q
        self.if_contamination  = if_contamination
        self.kmeans            = None
        self.iforest           = None
        self.km_threshold      = None

    # ── Training ───────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "ExpenseAnomalyDetector":
        """Fit both models on the (scaled) feature matrix."""
        # K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=RANDOM_SEED,
            n_init=10,
        )
        self.kmeans.fit(X)

        # Compute centroid distances and derive threshold
        distances = self._centroid_distances(X)
        self.km_threshold = np.quantile(distances, self.km_threshold_q)

        # Isolation Forest
        self.iforest = IsolationForest(
            contamination=self.if_contamination,
            random_state=RANDOM_SEED,
            n_estimators=200,
        )
        self.iforest.fit(X)

        return self

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> pd.DataFrame:
        """
        Run both detectors and return a results DataFrame.

        Returns a DataFrame with columns:
            cluster          : K-Means cluster label
            centroid_dist    : distance from assigned centroid
            km_flag          : True if K-Means flags as anomaly
            if_score         : Isolation Forest anomaly score (lower = more anomalous)
            if_flag          : True if Isolation Forest flags as anomaly
            anomaly          : True if BOTH methods agree (ensemble vote)
        """
        distances = self._centroid_distances(X)
        clusters  = self.kmeans.predict(X)
        km_flags  = distances > self.km_threshold

        if_preds  = self.iforest.predict(X)          # -1 = anomaly, +1 = normal
        if_scores = self.iforest.score_samples(X)    # lower = more anomalous
        if_flags  = if_preds == -1

        ensemble  = km_flags & if_flags              # both must agree

        return pd.DataFrame({
            "cluster":       clusters,
            "centroid_dist": np.round(distances, 4),
            "km_flag":       km_flags,
            "if_score":      np.round(if_scores, 4),
            "if_flag":       if_flags,
            "anomaly":       ensemble,
        })

    # ── Evaluation ─────────────────────────────────────────────────────────

    def evaluate(self, results: pd.DataFrame, y_true: pd.Series) -> dict:
        """
        Compare detected anomalies against ground-truth labels.

        Parameters
        ----------
        results : DataFrame returned by predict()
        y_true  : boolean Series (True = genuine anomaly)

        Returns
        -------
        dict with precision, recall, f1, confusion_matrix, report string
        """
        y_pred = results["anomaly"].astype(int)
        y_true_int = y_true.astype(int)

        cm     = confusion_matrix(y_true_int, y_pred)
        report = classification_report(y_true_int, y_pred, target_names=["Normal", "Anomaly"])

        return {
            "confusion_matrix": cm,
            "report":           report,
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    def _centroid_distances(self, X: np.ndarray) -> np.ndarray:
        """Euclidean distance from each point to its nearest centroid."""
        clusters   = self.kmeans.predict(X)
        centroids  = self.kmeans.cluster_centers_
        diffs      = X - centroids[clusters]
        return np.linalg.norm(diffs, axis=1)
