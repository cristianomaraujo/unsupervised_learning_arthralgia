# -*- coding: utf-8 -*-
"""arthralgia_clinical_patterns_unsupervised_analysis.ipynb

"""

!pip -q install openpyxl statsmodels

import warnings
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import orthogonal_procrustes
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

warnings.filterwarnings("ignore")

SEED = 40
np.random.seed(SEED)
PARALLEL_B = 500
STABILITY_B = 200
BOOTSTRAP_B = 1000
MAX_AXES_TO_TEST = 10

DISCOVERY_VARS = [
    "Pain_duration", "Pain_intensity", "Crepitus", "Myalgia",
    "TMD_headache", "Oral_behaviors"
]
NUMERIC_VARS = ["Pain_duration", "Pain_intensity"]
CATEGORICAL_VARS = ["Crepitus", "Myalgia", "TMD_headache", "Oral_behaviors"]

"""## Importação dos dados e correspondência das variáveis"""

candidate_paths = [
    Path("dataset3.xlsx"), Path("/content/dataset3.xlsx"),
    Path("dataset(20260922-183643).xlsx"),
    Path("/workspace/scratch/2f30f118185a/upload/dataset(20260922-183643).xlsx")
]
data_path = next((p for p in candidate_paths if p.exists()), None)
if data_path is None:
    try:
        from google.colab import files
        uploaded = files.upload()
        data_path = Path(next(iter(uploaded)))
    except Exception as exc:
        raise FileNotFoundError("Faça o upload do arquivo dataset3.xlsx antes de executar o notebook.") from exc

excel_file = pd.ExcelFile(data_path)
preferred_sheet = "Planilha1" if "Planilha1" in excel_file.sheet_names else excel_file.sheet_names[0]
raw = pd.read_excel(data_path, sheet_name=preferred_sheet)
print(f"Carregados {len(raw)} pacientes de {data_path} | aba: {preferred_sheet}")

COL = {
    "ID": "nº identificação panorâmica", "Sex": "sexo", "Age": "idade",
    "Pain_duration": "início da dor (não = 0 aguda = 1 crônica = 2)",
    "Pain_intensity": "intensidade da dor (VAS 0 a 10)",
    "Crepitus": "crepitação (não = 0 sim = 1)",
    "Myalgia": "mialgia (não = 0 sim = 1)",
    "TMD_headache": "cefaleia artribuída à DTM (não = 0 sim = 1)",
    "Oral_behaviors": "presença de bruxismo (não = 0 sim = 1)",
    "Arthralgia_R": "artralgia Direita (não = 0 sim = 1)",
    "Arthralgia_L": "artralgia Esquerda (não = 0 sim = 1)",
}
missing = [v for v in COL.values() if v not in raw.columns]
if missing:
    raise ValueError(f"Required columns not found: {missing}")

df = raw[list(COL.values())].rename(columns={v: k for k, v in COL.items()}).copy()
for c in [c for c in df.columns if c != "Sex"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
if df["ID"].duplicated().any():
    raise ValueError("Patient IDs are not unique in the wide sheet.")

df["Arthralgia_count"] = df[["Arthralgia_R", "Arthralgia_L"]].sum(axis=1, min_count=2)
df["Arthralgia_pattern"] = df["Arthralgia_count"].map({0: "Absent", 1: "Unilateral", 2: "Bilateral"})
df["Any_arthralgia"] = (df["Arthralgia_count"] >= 1).astype(float)
df["Bilateral_arthralgia"] = (df["Arthralgia_count"] == 2).astype(float)
df.loc[df["Arthralgia_count"].isna(), ["Any_arthralgia", "Bilateral_arthralgia"]] = np.nan

analysis_data = df.dropna(subset=["ID"]).reset_index(drop=True)
if not analysis_data["Pain_duration"].dropna().isin([0, 1, 2]).all():
    raise ValueError("Pain_duration deve conter somente 0=ausente, 1=aguda (<6 meses) ou 2=crônica (≥6 meses).")
for v in ["Crepitus", "Myalgia", "TMD_headache", "Oral_behaviors", "Arthralgia_R", "Arthralgia_L"]:
    if not analysis_data[v].dropna().isin([0, 1]).all():
        raise ValueError(f"{v} deve conter somente 0 ou 1.")
if not analysis_data["Pain_intensity"].dropna().between(0, 10).all():
    raise ValueError("Pain_intensity deve variar de 0 a 10.")
display(analysis_data["Arthralgia_pattern"].value_counts(dropna=False).to_frame("n"))

"""## Controle de qualidade das variáveis de descoberta"""

qc_rows = []
for v in DISCOVERY_VARS:
    vc = analysis_data[v].value_counts(normalize=True, dropna=True)
    qc_rows.append({
        "Variable": v, "Missing_n": int(analysis_data[v].isna().sum()),
        "Missing_percent": 100 * analysis_data[v].isna().mean(),
        "Unique_nonmissing": int(analysis_data[v].nunique(dropna=True)),
        "Most_common_percent": 100 * vc.iloc[0] if len(vc) else np.nan,
    })
qc_table = pd.DataFrame(qc_rows)
display(qc_table.round(2))
if (qc_table["Missing_percent"] > 30).any():
    raise ValueError("At least one discovery variable has >30% missingness.")
active_discovery = qc_table.loc[qc_table["Unique_nonmissing"] > 1, "Variable"].tolist()
active_numeric = [v for v in NUMERIC_VARS if v in active_discovery]
active_categorical = [v for v in CATEGORICAL_VARS if v in active_discovery]

"""## Distância de Gower e PCoA

A correção de Lingoes é aplicada quando autovalores negativos indicam que a matriz de Gower não é completamente euclidiana. A retenção dos eixos baseia-se na comparação dos autovalores brutos observados com bancos de referência permutados independentemente. As coordenadas corrigidas são utilizadas nos escores dos participantes e nas análises subsequentes.
"""

def gower_distance(data, numeric, categorical):
    n = len(data)
    numerator = np.zeros((n, n), float)
    denominator = np.zeros((n, n), float)
    for v in numeric:
        x = pd.to_numeric(data[v], errors="coerce").to_numpy(float)
        valid = np.isfinite(x)
        rg = np.ptp(x[valid]) if valid.any() else np.nan
        if not np.isfinite(rg) or rg == 0:
            continue
        pair = valid[:, None] & valid[None, :]
        numerator += np.where(pair, np.abs(x[:, None] - x[None, :]) / rg, 0)
        denominator += pair
    for v in categorical:
        x = data[v].astype("object").to_numpy()
        valid = pd.notna(x)
        pair = valid[:, None] & valid[None, :]
        numerator += np.where(pair, (x[:, None] != x[None, :]).astype(float), 0)
        denominator += pair
    D = numerator / denominator
    np.fill_diagonal(D, 0)
    if np.isnan(D).any():
        raise ValueError("At least one pair shares no observed discovery variable.")
    return D


def pcoa_raw(D):
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def pcoa_lingoes(D):
    raw_eigenvalues, _ = pcoa_raw(D)
    correction = max(0.0, -float(raw_eigenvalues.min()))
    if correction > 0:
        corrected_squared = D ** 2 + 2 * correction
        np.fill_diagonal(corrected_squared, 0.0)
        corrected_D = np.sqrt(np.maximum(corrected_squared, 0))
    else:
        corrected_D = D.copy()
    eigenvalues, eigenvectors = pcoa_raw(corrected_D)
    positive = eigenvalues > 1e-10
    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]
    coordinates = eigenvectors * np.sqrt(eigenvalues)
    return raw_eigenvalues, eigenvalues, coordinates, correction, corrected_D


D = gower_distance(analysis_data, active_numeric, active_categorical)
raw_eigenvalues, corrected_eigenvalues, coordinates_all, lingoes_constant, corrected_D = pcoa_lingoes(D)
positive_raw = raw_eigenvalues[raw_eigenvalues > 1e-10]
negative_raw = raw_eigenvalues[raw_eigenvalues < -1e-10]
negative_inertia_ratio = abs(negative_raw.sum()) / positive_raw.sum()

print(f"Lingoes constant: {lingoes_constant:.6f}")
print(f"Negative/positive raw inertia ratio: {negative_inertia_ratio:.4f}")

"""## Análise paralela e comparação broken-stick"""

def permute_columns(data, variables, rng):
    ref = data[variables].copy().reset_index(drop=True)
    for v in variables:
        ref[v] = rng.permutation(ref[v].to_numpy())
    return ref

rng = np.random.default_rng(SEED)
reference_eigenvalues = []
for b in range(PARALLEL_B):
    ref = permute_columns(analysis_data, active_discovery, rng)
    ref_D = gower_distance(ref, active_numeric, active_categorical)
    ref_raw, _ = pcoa_raw(ref_D)
    arr = np.zeros(MAX_AXES_TO_TEST)
    positive = ref_raw[ref_raw > 1e-10]
    arr[:min(MAX_AXES_TO_TEST, len(positive))] = positive[:MAX_AXES_TO_TEST]
    reference_eigenvalues.append(arr)

reference_eigenvalues = np.asarray(reference_eigenvalues)
parallel_95 = np.percentile(reference_eigenvalues, 95, axis=0)
observed_top = np.zeros(MAX_AXES_TO_TEST)
observed_top[:min(MAX_AXES_TO_TEST, len(positive_raw))] = positive_raw[:MAX_AXES_TO_TEST]

parallel_retained = observed_top > parallel_95
n_retained = 0
for keep in parallel_retained:
    if keep:
        n_retained += 1
    else:
        break

raw_prop = positive_raw / positive_raw.sum()
m = len(positive_raw)
broken_stick = np.array([sum(1 / j for j in range(k, m + 1)) / m for k in range(1, m + 1)])

axis_selection = pd.DataFrame({
    "Axis": np.arange(1, MAX_AXES_TO_TEST + 1),
    "Observed_raw_eigenvalue": observed_top,
    "Parallel_95th_percentile": parallel_95,
    "Retained_parallel": parallel_retained,
    "Raw_positive_inertia_percent": 100 * raw_prop[:MAX_AXES_TO_TEST],
    "Broken_stick_percent": 100 * broken_stick[:MAX_AXES_TO_TEST],
})
display(axis_selection.round(4))

if n_retained == 0:
    raise RuntimeError("Parallel analysis retained no clinical dimension.")

coordinates = coordinates_all[:, :n_retained].copy()
axis_names = [f"Clinical_Dimension_{i+1}" for i in range(n_retained)]
for j, name in enumerate(axis_names):
    analysis_data[name] = coordinates[:, j]

# Orientação reprodutível: a Dimensão 1 aumenta com a intensidade da dor.
if stats.spearmanr(analysis_data[axis_names[0]], analysis_data["Pain_intensity"], nan_policy="omit").statistic < 0:
    analysis_data[axis_names[0]] *= -1
    coordinates[:, 0] *= -1

print("Dimensions retained by parallel analysis:", n_retained)

"""## Estabilidade por subamostragem

Em cada iteração, a PCoA é reajustada em 80% dos participantes. As coordenadas das subamostras são alinhadas às coordenadas correspondentes da amostra completa por rotação ortogonal de Procrustes. As correlações entre os escores quantificam a estabilidade.
"""

stability_rows = []
for b in range(STABILITY_B):
    rng = np.random.default_rng(SEED + b)
    idx = np.sort(rng.choice(len(analysis_data), size=int(np.ceil(.80 * len(analysis_data))), replace=False))
    sub = analysis_data.iloc[idx]
    sub_D = gower_distance(sub, active_numeric, active_categorical)
    _, _, sub_coordinates_all, _, _ = pcoa_lingoes(sub_D)
    q = min(n_retained, sub_coordinates_all.shape[1])
    reference = coordinates[idx, :q]
    candidate = sub_coordinates_all[:, :q]
    reference_centered = reference - reference.mean(axis=0)
    candidate_centered = candidate - candidate.mean(axis=0)
    rotation, _ = orthogonal_procrustes(candidate_centered, reference_centered)
    aligned = candidate_centered @ rotation
    for j in range(q):
        stability_rows.append({
            "Iteration": b + 1, "Axis": j + 1,
            "Score_correlation": stats.pearsonr(reference_centered[:, j], aligned[:, j]).statistic
        })

stability_iterations = pd.DataFrame(stability_rows)
stability_summary = stability_iterations.groupby("Axis")["Score_correlation"].agg(
    Mean="mean", Median="median",
    P025=lambda x: np.percentile(x, 2.5),
    P975=lambda x: np.percentile(x, 97.5)
).reset_index()
display(stability_summary.round(4))

"""## Caracterização clínica das dimensões retidas

As relações entre os escores e as variáveis de entrada são apresentadas apenas para interpretação descritiva. Como essas variáveis participaram da construção das dimensões, não são realizados testes de hipótese nem correção por multiplicidade nesta etapa. Intervalos de confiança bootstrap descrevem a incerteza das magnitudes.
"""

def correlation_ratio(categories, values):
    temp = pd.DataFrame({"category": categories, "value": values}).dropna()
    grand = temp["value"].mean()
    denominator = ((temp["value"] - grand) ** 2).sum()
    numerator = sum(len(g) * (g["value"].mean() - grand) ** 2 for _, g in temp.groupby("category"))
    return numerator / denominator if denominator > 0 else np.nan

def descriptive_effect(frame, axis, variable):
    if variable in NUMERIC_VARS:
        complete = frame[[axis, variable]].dropna()
        return stats.spearmanr(complete[axis], complete[variable]).statistic
    complete = frame[[axis, variable]].dropna()
    return correlation_ratio(complete[variable], complete[axis])

association_rows = []
rng_boot = np.random.default_rng(SEED + 10000)
for axis in axis_names:
    for v in active_discovery:
        effect = descriptive_effect(analysis_data, axis, v)
        boot = []
        for _ in range(BOOTSTRAP_B):
            idx = rng_boot.integers(0, len(analysis_data), len(analysis_data))
            value = descriptive_effect(analysis_data.iloc[idx], axis, v)
            if np.isfinite(value):
                boot.append(value)
        association_rows.append({
            "Axis": axis, "Variable": v,
            "Measure": "Spearman_rho" if v in NUMERIC_VARS else "Eta_squared",
            "Effect": effect,
            "Bootstrap_CI95_lower": np.percentile(boot, 2.5),
            "Bootstrap_CI95_upper": np.percentile(boot, 97.5),
        })

dimension_associations = pd.DataFrame(association_rows)
display(dimension_associations.round(4))

"""## Associação pós-hoc com a extensão da artralgia"""

# Descriptive monotonic relationship with the number of affected sides
arthralgia_axis = []
for axis in axis_names:
    rho, p = stats.spearmanr(analysis_data[axis], analysis_data["Arthralgia_count"], nan_policy="omit")
    arthralgia_axis.append({"Axis": axis, "Spearman_rho": rho, "p_value": p})
arthralgia_axis = pd.DataFrame(arthralgia_axis)
display(arthralgia_axis.round(4))

# Análise secundária no nível do paciente: ausência é a categoria de referência.
mn_data = analysis_data[["Arthralgia_pattern", "Age", "Sex"] + axis_names].dropna().copy()
mn_data["Outcome"] = pd.Categorical(
    mn_data["Arthralgia_pattern"], categories=["Absent", "Unilateral", "Bilateral"], ordered=True
).codes
mn_data["Sex_Male"] = (mn_data["Sex"].astype(str).str.upper() == "M").astype(int)
mn_data["Age_centered"] = mn_data["Age"] - mn_data["Age"].mean()
for axis in axis_names:
    mn_data[axis + "_z"] = (mn_data[axis] - mn_data[axis].mean()) / mn_data[axis].std(ddof=0)
mn_predictors = [a + "_z" for a in axis_names] + ["Age_centered", "Sex_Male"]
X_mn = sm.add_constant(mn_data[mn_predictors].astype(float), has_constant="add")
mn_model = sm.MNLogit(mn_data["Outcome"], X_mn).fit(method="newton", maxiter=200, disp=False)

mn_rows = []
for outcome_col in mn_model.params.columns:
    outcome_label = ["Unilateral vs absent", "Bilateral vs absent"][int(outcome_col)]
    for term in mn_model.params.index:
        beta = mn_model.params.loc[term, outcome_col]
        se = mn_model.bse.loc[term, outcome_col]
        p = mn_model.pvalues.loc[term, outcome_col]
        mn_rows.append({
            "Comparison": outcome_label, "Term": term, "Beta": beta, "SE": se,
            "p_value": p, "RRR": np.exp(beta),
            "RRR_CI95_lower": np.exp(beta - 1.96 * se),
            "RRR_CI95_upper": np.exp(beta + 1.96 * se),
        })
multinomial_results = pd.DataFrame(mn_rows)
mn_axis_mask = multinomial_results["Term"].str.startswith("Clinical_Dimension")
multinomial_results.loc[mn_axis_mask, "p_BH_axes_and_comparisons"] = sm.stats.multipletests(
    multinomial_results.loc[mn_axis_mask, "p_value"], method="fdr_bh"
)[1]
display(multinomial_results.round(4))

"""## Análise principal: artralgia específica por lado

Logistic GEE accounts for the correlation between the right and left sides of the same patient. Dimension scores are patient-level predictors; age, sex, and side are covariates.
"""

base_cols = ["ID", "Age", "Sex"] + axis_names
right = analysis_data[base_cols + ["Arthralgia_R"]].copy().rename(columns={"Arthralgia_R": "Arthralgia"})
right["Side"] = "Right"
left = analysis_data[base_cols + ["Arthralgia_L"]].copy().rename(columns={"Arthralgia_L": "Arthralgia"})
left["Side"] = "Left"
side_data = pd.concat([right, left], ignore_index=True).dropna()
side_data["Sex_Male"] = (side_data["Sex"].astype(str).str.upper() == "M").astype(int)
side_data["Side_Left"] = (side_data["Side"] == "Left").astype(int)
side_data["Age_centered"] = side_data["Age"] - side_data["Age"].mean()
for axis in axis_names:
    mu, sd = analysis_data[axis].mean(), analysis_data[axis].std(ddof=0)
    side_data[axis + "_z"] = (side_data[axis] - mu) / sd

gee_predictors = [a + "_z" for a in axis_names] + ["Age_centered", "Sex_Male", "Side_Left"]
X_gee = sm.add_constant(side_data[gee_predictors].astype(float), has_constant="add")
gee = GEE(
    side_data["Arthralgia"].astype(float), X_gee,
    groups=side_data["ID"], family=Binomial(), cov_struct=Exchangeable()
).fit()
ci = gee.conf_int()
gee_arthralgia_results = pd.DataFrame({
    "Term": gee.params.index, "Beta": gee.params.values, "SE": gee.bse.values,
    "p_value": gee.pvalues.values, "OR": np.exp(gee.params.values),
    "OR_CI95_lower": np.exp(ci[0].values), "OR_CI95_upper": np.exp(ci[1].values)
})
axis_mask = gee_arthralgia_results["Term"].str.startswith("Clinical_Dimension")
gee_arthralgia_results.loc[axis_mask, "p_BH_axes"] = sm.stats.multipletests(
    gee_arthralgia_results.loc[axis_mask, "p_value"], method="fdr_bh"
)[1]
display(gee_arthralgia_results.round(4))

"""## Análise de sensibilidade sem crepitação

A descoberta é repetida sem a crepitação para verificar se o padrão clínico e sua associação com a artralgia dependem do único sinal articular incluído. A análise compara o número de eixos retidos, a correlação entre os primeiros escores e o efeito da dimensão alternativa na GEE principal.
"""

sens_vars = [v for v in active_discovery if v != "Crepitus"]
sens_numeric = [v for v in NUMERIC_VARS if v in sens_vars]
sens_categorical = [v for v in CATEGORICAL_VARS if v in sens_vars]

sens_D = gower_distance(analysis_data, sens_numeric, sens_categorical)
sens_raw, sens_corr_eig, sens_coords_all, sens_lingoes, _ = pcoa_lingoes(sens_D)
sens_positive = sens_raw[sens_raw > 1e-10]

rng = np.random.default_rng(SEED + 50000)
sens_reference = []
for b in range(PARALLEL_B):
    ref = permute_columns(analysis_data, sens_vars, rng)
    ref_D = gower_distance(ref, sens_numeric, sens_categorical)
    ref_raw, _ = pcoa_raw(ref_D)
    arr = np.zeros(MAX_AXES_TO_TEST)
    pos = ref_raw[ref_raw > 1e-10]
    arr[:min(MAX_AXES_TO_TEST, len(pos))] = pos[:MAX_AXES_TO_TEST]
    sens_reference.append(arr)
sens_reference = np.asarray(sens_reference)
sens_parallel_95 = np.percentile(sens_reference, 95, axis=0)
sens_observed = np.zeros(MAX_AXES_TO_TEST)
sens_observed[:min(MAX_AXES_TO_TEST, len(sens_positive))] = sens_positive[:MAX_AXES_TO_TEST]
sens_flags = sens_observed > sens_parallel_95
sens_n_retained = 0
for keep in sens_flags:
    if keep:
        sens_n_retained += 1
    else:
        break
if sens_n_retained == 0:
    raise RuntimeError("A análise de sensibilidade não reteve dimensão clínica.")

sens_score = sens_coords_all[:, 0].copy()
if stats.spearmanr(sens_score, analysis_data["Pain_intensity"], nan_policy="omit").statistic < 0:
    sens_score *= -1
analysis_data["Sensitivity_Dimension_1_no_crepitus"] = sens_score
score_rho = stats.spearmanr(
    analysis_data[axis_names[0]], analysis_data["Sensitivity_Dimension_1_no_crepitus"], nan_policy="omit"
).statistic

sens_side = side_data[["ID", "Arthralgia", "Age_centered", "Sex_Male", "Side_Left"]].copy()
sens_map = analysis_data.set_index("ID")["Sensitivity_Dimension_1_no_crepitus"]
sens_side["Sensitivity_Dimension_1_z"] = sens_side["ID"].map(sens_map)
sens_side["Sensitivity_Dimension_1_z"] = (
    sens_side["Sensitivity_Dimension_1_z"] - sens_side["Sensitivity_Dimension_1_z"].mean()
) / sens_side["Sensitivity_Dimension_1_z"].std(ddof=0)
X_sens = sm.add_constant(
    sens_side[["Sensitivity_Dimension_1_z", "Age_centered", "Sex_Male", "Side_Left"]].astype(float),
    has_constant="add"
)
sens_gee = GEE(
    sens_side["Arthralgia"].astype(float), X_sens, groups=sens_side["ID"],
    family=Binomial(), cov_struct=Exchangeable()
).fit()
sens_ci = sens_gee.conf_int().loc["Sensitivity_Dimension_1_z"]
sensitivity_summary = pd.DataFrame([{
    "Variables": ", ".join(sens_vars),
    "Dimensions_retained": sens_n_retained,
    "Lingoes_constant": sens_lingoes,
    "Spearman_with_primary_Dimension_1": score_rho,
    "GEE_OR_per_SD": np.exp(sens_gee.params["Sensitivity_Dimension_1_z"]),
    "GEE_OR_CI95_lower": np.exp(sens_ci.iloc[0]),
    "GEE_OR_CI95_upper": np.exp(sens_ci.iloc[1]),
    "GEE_p_value": sens_gee.pvalues["Sensitivity_Dimension_1_z"],
}])
display(sensitivity_summary.round(4))

"""## Figuras"""

FIGURE_DIR = Path("arthralgia_clinical_patterns_figures")
FIGURE_DIR.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

def savefig(name):
    plt.tight_layout(); plt.savefig(FIGURE_DIR / name, dpi=600, bbox_inches="tight", facecolor="white"); plt.show(); plt.close()

# Parallel analysis
plt.figure(figsize=(8, 5.5))
xx = np.arange(1, MAX_AXES_TO_TEST + 1)
plt.plot(xx, observed_top, marker="o", label="Observado")
plt.plot(xx, parallel_95, marker="o", linestyle="--", label="Percentil 95 das permutações")
plt.xlabel("Eixo da PCoA"); plt.ylabel("Autovalor bruto"); plt.title("Análise paralela")
plt.xticks(xx); plt.grid(alpha=.2); plt.legend(frameon=False)
savefig("01_parallel_analysis.png")

# Clinical characterization
display_labels = {
    "Pain_duration": "Temporalidade da dor", "Pain_intensity": "Intensidade da dor",
    "Crepitus": "Crepitação", "Myalgia": "Mialgia",
    "TMD_headache": "Cefaleia atribuída à DTM",
    "Oral_behaviors": "Comportamentos orais autorreferidos"
}
pivot = dimension_associations.pivot(index="Variable", columns="Axis", values="Effect")
pivot.index = [display_labels.get(v, v) for v in pivot.index]
fig, ax = plt.subplots(figsize=(max(5, 2.2 * n_retained), 5.8))
im = ax.imshow(pivot.to_numpy(), cmap="Blues", aspect="auto", vmin=0)
ax.set_xticks(range(len(pivot.columns)), pivot.columns)
ax.set_yticks(range(len(pivot.index)), pivot.index)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        ax.text(j, i, f"{pivot.iloc[i,j]:.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax, label="Magnitude da associação")
plt.title("Caracterização clínica das dimensões retidas")
savefig("02_dimension_associations.png")

# Dimension 1 by arthralgia pattern
order = ["Absent", "Unilateral", "Bilateral"]
order_labels = ["Ausente", "Unilateral", "Bilateral"]
groups = [analysis_data.loc[analysis_data["Arthralgia_pattern"] == g, axis_names[0]].dropna() for g in order]
plt.figure(figsize=(7.5, 5.5))
plt.boxplot(groups, labels=order_labels, showfliers=False)
for i, values in enumerate(groups, start=1):
    jitter = np.random.default_rng(SEED+i).normal(i, .04, len(values))
    plt.scatter(jitter, values, s=18, alpha=.35)
plt.xlabel("Padrão de artralgia"); plt.ylabel("Escore da Dimensão Clínica 1")
plt.title("Dimensão clínica segundo a extensão da artralgia"); plt.grid(axis="y", alpha=.15)
savefig("03_dimension1_arthralgia.png")

"""## Exportação"""

output_file = "arthralgia_clinical_patterns_results.xlsx"
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    analysis_data.to_excel(writer, sheet_name="Analysis_Data", index=False)
    qc_table.to_excel(writer, sheet_name="Discovery_QC", index=False)
    pd.DataFrame(D, index=analysis_data["ID"], columns=analysis_data["ID"]).to_excel(writer, sheet_name="Gower_Distance")
    axis_selection.to_excel(writer, sheet_name="Axis_Selection", index=False)
    pd.DataFrame(reference_eigenvalues, columns=[f"Axis_{i}" for i in range(1, MAX_AXES_TO_TEST+1)]).to_excel(writer, sheet_name="Parallel_Iterations", index=False)
    stability_summary.to_excel(writer, sheet_name="Axis_Stability", index=False)
    stability_iterations.to_excel(writer, sheet_name="Stability_Iterations", index=False)
    dimension_associations.to_excel(writer, sheet_name="Dimension_Associations", index=False)
    arthralgia_axis.to_excel(writer, sheet_name="Arthralgia_Correlation", index=False)
    multinomial_results.to_excel(writer, sheet_name="Arthralgia_Multinomial", index=False)
    gee_arthralgia_results.to_excel(writer, sheet_name="Arthralgia_GEE", index=False)
    sensitivity_summary.to_excel(writer, sheet_name="Sensitivity_No_Crepitus", index=False)
    pd.DataFrame([{
        "N": len(analysis_data), "Dimensions_retained": n_retained,
        "Lingoes_constant": lingoes_constant, "Negative_inertia_ratio": negative_inertia_ratio,
        "Parallel_iterations": PARALLEL_B, "Stability_iterations": STABILITY_B,
        "Primary_representation": "Gower PCoA with Lingoes correction",
        "Primary_outcome_model": "Side-specific logistic GEE",
        "Discovery_variables": ", ".join(active_discovery),
        "Arthralgia_used_in_discovery": "No"
    }]).to_excel(writer, sheet_name="Decision_Summary", index=False)

figure_zip = shutil.make_archive("arthralgia_clinical_patterns_figures", "zip", FIGURE_DIR)
print("Saved:", output_file); print("Figures:", figure_zip)
try:
    from google.colab import files
    files.download(output_file); files.download(figure_zip)
except Exception:
    pass

# @title
# ============================================================
# SCIENTIFIC FIGURES — CLINICAL PATTERNS AND ARTHRALGIA
# ============================================================

# If necessary in Google Colab:
# !pip install openpyxl seaborn scipy

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy.stats import chi2, spearmanr


# ============================================================
# 1. SETTINGS
# ============================================================

# Change this path only if the file is stored elsewhere
FILE = Path("arthralgia_clinical_patterns_results.xlsx")

OUTPUT_FOLDER = Path("arthralgia_figures")
OUTPUT_FOLDER.mkdir(exist_ok=True)

DPI = 600
SEED = 40
rng = np.random.default_rng(SEED)

# Colorblind-friendly palette
ARTHRALGIA_COLORS = {
    "Absent": "#4C78A8",
    "Unilateral": "#E6A23C",
    "Bilateral": "#C44E52"
}

ARTHRALGIA_ORDER = [
    "Absent",
    "Unilateral",
    "Bilateral"
]

DIMENSION_1_COLOR = "#B84A3A"
DIMENSION_2_COLOR = "#3E6F9E"
NEUTRAL_COLOR = "#64748B"
GRID_COLOR = "#D9E0E7"
TEXT_COLOR = "#243342"

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.titleweight": "bold",
    "axes.labelcolor": TEXT_COLOR,
    "axes.edgecolor": "#9AA8B5",
    "axes.linewidth": 0.8,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight"
})

sns.set_theme(
    style="whitegrid",
    context="paper"
)


# ============================================================
# 2. READ DATA
# ============================================================

if not FILE.exists():
    raise FileNotFoundError(
        f"File not found: {FILE.resolve()}\n"
        "Update the FILE variable at the beginning of this block."
    )

analysis_data = pd.read_excel(
    FILE,
    sheet_name="Analysis_Data"
)

axis_selection = pd.read_excel(
    FILE,
    sheet_name="Axis_Selection"
)

dimension_associations = pd.read_excel(
    FILE,
    sheet_name="Dimension_Associations"
)

multinomial = pd.read_excel(
    FILE,
    sheet_name="Arthralgia_Multinomial"
)

gee = pd.read_excel(
    FILE,
    sheet_name="Arthralgia_GEE"
)

sensitivity = pd.read_excel(
    FILE,
    sheet_name="Sensitivity_No_Crepitus"
)

analysis_data["Arthralgia_pattern"] = pd.Categorical(
    analysis_data["Arthralgia_pattern"],
    categories=ARTHRALGIA_ORDER,
    ordered=True
)

dimension_1_inertia = axis_selection.loc[
    axis_selection["Axis"] == 1,
    "Raw_positive_inertia_percent"
].iloc[0]

dimension_2_inertia = axis_selection.loc[
    axis_selection["Axis"] == 2,
    "Raw_positive_inertia_percent"
].iloc[0]


# ============================================================
# 3. AUXILIARY FUNCTIONS
# ============================================================

def format_axis(ax):
    """Apply a consistent visual style to an axis."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="both",
        color=GRID_COLOR,
        linewidth=0.6,
        alpha=0.55
    )

    ax.set_axisbelow(True)


def add_panel_letter(ax, letter):
    """Add a panel identification letter."""

    ax.text(
        -0.12,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left",
        color="#17212B"
    )


def add_concentration_ellipse(
    ax,
    x,
    y,
    color,
    level=0.80,
    linewidth=1.8,
    alpha=0.12
):
    """
    Add a descriptive bivariate concentration ellipse.

    The ellipse does not represent a cluster or a formal
    confidence region for group classification.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return

    covariance = np.cov(x, y)
    center = np.array([
        np.mean(x),
        np.mean(y)
    ])

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = eigenvalues.argsort()[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    eigenvalues = np.maximum(eigenvalues, 0)

    angle = np.degrees(
        np.arctan2(
            eigenvectors[1, 0],
            eigenvectors[0, 0]
        )
    )

    scale = np.sqrt(
        chi2.ppf(level, df=2)
    )

    width = (
        2 *
        scale *
        np.sqrt(eigenvalues[0])
    )

    height = (
        2 *
        scale *
        np.sqrt(eigenvalues[1])
    )

    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=1
    )

    ax.add_patch(ellipse)

    # Group centroid
    ax.scatter(
        center[0],
        center[1],
        s=95,
        marker="X",
        color=color,
        edgecolor="white",
        linewidth=1.2,
        zorder=5
    )


def save_figure(fig, filename):
    """
    Save a figure in PNG, TIFF, and PDF formats.
    """

    fig.savefig(
        OUTPUT_FOLDER / f"{filename}.png",
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        OUTPUT_FOLDER / f"{filename}.tiff",
        dpi=DPI,
        bbox_inches="tight",
        pil_kwargs={
            "compression": "tiff_lzw"
        }
    )

    fig.savefig(
        OUTPUT_FOLDER / f"{filename}.pdf",
        bbox_inches="tight"
    )


def plot_dimension_associations(
    ax,
    data,
    dimension,
    color,
    title
):
    """
    Plot the clinical characterization of a retained dimension.

    Spearman rho may assume positive or negative values.
    Eta-squared ranges from zero to one and has no direction.
    """

    subset = data.loc[
        data["Axis"] == dimension
    ].copy()

    variable_order = [
        "Pain_duration",
        "Pain_intensity",
        "Crepitus",
        "Myalgia",
        "TMD_headache",
        "Oral_behaviors"
    ]

    variable_labels = {
        "Pain_duration": "Pain duration",
        "Pain_intensity": "Pain intensity",
        "Crepitus": "Crepitus",
        "Myalgia": "Myalgia",
        "TMD_headache": "Headache attributed to TMD",
        "Oral_behaviors": "Self-reported oral behaviors"
    }

    subset["Variable"] = pd.Categorical(
        subset["Variable"],
        categories=variable_order[::-1],
        ordered=True
    )

    subset = subset.sort_values(
        "Variable"
    )

    y_positions = np.arange(
        len(subset)
    )

    # Lines extending from zero to each estimate
    for y_position, effect in zip(
        y_positions,
        subset["Effect"]
    ):
        ax.plot(
            [0, effect],
            [y_position, y_position],
            color="#CBD5DF",
            linewidth=2.4,
            solid_capstyle="round",
            zorder=1
        )

    for y_position, (_, row) in zip(
        y_positions,
        subset.iterrows()
    ):
        measure = row["Measure"]

        marker = (
            "o"
            if measure == "Spearman_rho"
            else "s"
        )

        ax.errorbar(
            row["Effect"],
            y_position,
            xerr=[[
                row["Effect"] -
                row["Bootstrap_CI95_lower"]
            ], [
                row["Bootstrap_CI95_upper"] -
                row["Effect"]
            ]],
            fmt=marker,
            markersize=7.5,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
            zorder=3
        )

        symbol = (
            "ρ"
            if measure == "Spearman_rho"
            else "η²"
        )

        ax.text(
            row["Bootstrap_CI95_upper"] + 0.025,
            y_position,
            f"{symbol} = {row['Effect']:.2f}",
            va="center",
            fontsize=8.5,
            color=TEXT_COLOR
        )

    ax.axvline(
        0,
        color="#7A8793",
        linewidth=0.9,
        linestyle="--"
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels([
        variable_labels[variable]
        for variable in subset["Variable"]
    ])

    ax.set_xlim(
        -0.58,
        0.87
    )

    ax.set_xlabel(
        "Magnitude of association with the clinical dimension"
    )

    ax.set_title(
        title,
        loc="left",
        pad=10
    )

    format_axis(ax)
    ax.grid(axis="y", visible=False)


def raincloud_plot(
    ax,
    data,
    variable,
    title,
    dimension_color
):
    """
    Approximate raincloud plot combining a violin plot,
    boxplot, individual observations, and median markers.
    """

    plot_data = data.dropna(
        subset=[
            "Arthralgia_pattern",
            variable
        ]
    ).copy()

    sns.violinplot(
        data=plot_data,
        x="Arthralgia_pattern",
        y=variable,
        order=ARTHRALGIA_ORDER,
        palette=ARTHRALGIA_COLORS,
        hue="Arthralgia_pattern",
        hue_order=ARTHRALGIA_ORDER,
        legend=False,
        inner=None,
        cut=0,
        linewidth=0,
        saturation=0.85,
        ax=ax
    )

    # Apply transparency to violin objects
    for collection in ax.collections:
        try:
            collection.set_alpha(0.28)
        except Exception:
            pass

    sns.boxplot(
        data=plot_data,
        x="Arthralgia_pattern",
        y=variable,
        order=ARTHRALGIA_ORDER,
        palette=ARTHRALGIA_COLORS,
        hue="Arthralgia_pattern",
        hue_order=ARTHRALGIA_ORDER,
        legend=False,
        width=0.24,
        showfliers=False,
        boxprops={
            "edgecolor": "#344454",
            "linewidth": 1.0,
            "zorder": 4
        },
        whiskerprops={
            "color": "#344454",
            "linewidth": 1.0
        },
        capprops={
            "color": "#344454",
            "linewidth": 1.0
        },
        medianprops={
            "color": "white",
            "linewidth": 2.0
        },
        ax=ax
    )

    sns.stripplot(
        data=plot_data,
        x="Arthralgia_pattern",
        y=variable,
        order=ARTHRALGIA_ORDER,
        palette=ARTHRALGIA_COLORS,
        hue="Arthralgia_pattern",
        hue_order=ARTHRALGIA_ORDER,
        legend=False,
        jitter=0.20,
        size=3.2,
        alpha=0.48,
        edgecolor="white",
        linewidth=0.25,
        ax=ax
    )

    medians = (
        plot_data
        .groupby(
            "Arthralgia_pattern",
            observed=True
        )[variable]
        .median()
        .reindex(ARTHRALGIA_ORDER)
    )

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    for i, group in enumerate(
        ARTHRALGIA_ORDER
    ):
        n_group = (
            plot_data["Arthralgia_pattern"] == group
        ).sum()

        ax.text(
            i,
            y_min + 0.02 * y_range,
            f"n = {n_group}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#52606D"
        )

        if pd.notna(medians.loc[group]):
            ax.scatter(
                i,
                medians.loc[group],
                marker="D",
                s=28,
                color=dimension_color,
                edgecolor="white",
                linewidth=0.8,
                zorder=8
            )

    ax.set_xlabel(
        "Arthralgia pattern"
    )

    ax.set_ylabel(
        "Clinical dimension score"
    )

    ax.set_title(
        title,
        loc="left",
        pad=10
    )

    format_axis(ax)
    ax.grid(axis="x", visible=False)


# ============================================================
# 4. FIGURE 1
# UNSUPERVISED ORGANIZATION OF CLINICAL MANIFESTATIONS
# ============================================================

fig = plt.figure(
    figsize=(14.8, 8.8)
)
grid = fig.add_gridspec(
    nrows=2,
    ncols=2,
    width_ratios=[1.15, 1],
    height_ratios=[1, 1],
    wspace=0.44,
    hspace=0.48
)

ax_a = fig.add_subplot(
    grid[:, 0]
)

ax_b = fig.add_subplot(
    grid[0, 1]
)

ax_c = fig.add_subplot(
    grid[1, 1]
)

# Panel A: two-dimensional clinical map
for group in ARTHRALGIA_ORDER:

    subset = analysis_data.loc[
        analysis_data["Arthralgia_pattern"] == group
    ].copy()

    ax_a.scatter(
        subset["Clinical_Dimension_1"],
        subset["Clinical_Dimension_2"],
        s=45,
        color=ARTHRALGIA_COLORS[group],
        alpha=0.67,
        edgecolor="white",
        linewidth=0.55,
        label=f"{group} (n = {len(subset)})",
        zorder=3
    )

    add_concentration_ellipse(
        ax_a,
        subset["Clinical_Dimension_1"],
        subset["Clinical_Dimension_2"],
        color=ARTHRALGIA_COLORS[group],
        level=0.80
    )

ax_a.axhline(
    0,
    color="#A8B3BD",
    linewidth=0.8,
    linestyle="--"
)

ax_a.axvline(
    0,
    color="#A8B3BD",
    linewidth=0.8,
    linestyle="--"
)

ax_a.set_xlabel(
    f"Dimension 1: pain–muscle burden "
    f"({dimension_1_inertia:.1f}% of positive inertia)"
)

ax_a.set_ylabel(
    f"Dimension 2: crepitus-dominated pattern "
    f"({dimension_2_inertia:.1f}% of positive inertia)"
)

ax_a.set_title(
    "Clinical organization of patients",
    loc="left",
    pad=12
)

ax_a.legend(
    title="Arthralgia",
    frameon=False,
    loc="best",
    fontsize=9,
    title_fontsize=9.5
)

format_axis(ax_a)
add_panel_letter(ax_a, "A")

# Panels B and C: clinical characterization
plot_dimension_associations(
    ax_b,
    dimension_associations,
    dimension="Clinical_Dimension_1",
    color=DIMENSION_1_COLOR,
    title="Clinical characterization of Dimension 1"
)

plot_dimension_associations(
    ax_c,
    dimension_associations,
    dimension="Clinical_Dimension_2",
    color=DIMENSION_2_COLOR,
    title="Clinical characterization of Dimension 2"
)

add_panel_letter(ax_b, "B")
add_panel_letter(ax_c, "C")

fig.suptitle(
    "Unsupervised organization of TMD-related clinical manifestations",
    x=0.055,
    y=1.01,
    ha="left",
    fontsize=15,
    fontweight="bold",
    color="#17212B"
)

fig.text(
    0.055,
    -0.015,
    "Ellipses represent descriptive concentration regions and do not indicate "
    "discrete clusters. Arthralgia was not included in the construction of the "
    "clinical dimensions. Error bars represent 95% bootstrap confidence intervals.",
    ha="left",
    fontsize=8.5,
    color="#536170"
)

save_figure(
    fig,
    "Figure_1_clinical_organization"
)

plt.show()


# ============================================================
# 5. PREPARE DATA FOR THE FOREST PLOT
# ============================================================

# GEE: side-specific arthralgia
forest_gee = gee.loc[
    gee["Term"].isin([
        "Clinical_Dimension_1_z",
        "Clinical_Dimension_2_z"
    ])
].copy()

forest_gee["Model"] = (
    "GEE: side-specific arthralgia"
)

forest_gee["Comparison_label"] = (
    "Present vs. absent"
)

forest_gee["Estimate"] = forest_gee["OR"]

forest_gee["CI_lower"] = (
    forest_gee["OR_CI95_lower"]
)

forest_gee["CI_upper"] = (
    forest_gee["OR_CI95_upper"]
)

forest_gee["Adjusted_p"] = (
    forest_gee["p_BH_axes"]
)

# Multinomial model: patient-level arthralgia pattern
forest_multinomial = multinomial.loc[
    multinomial["Term"].isin([
        "Clinical_Dimension_1_z",
        "Clinical_Dimension_2_z"
    ])
].copy()

forest_multinomial["Model"] = (
    "Multinomial: patient-level pattern"
)

forest_multinomial["Comparison_label"] = (
    forest_multinomial["Comparison"].replace({
        "Unilateral vs absent":
            "Unilateral vs. absent",
        "Bilateral vs absent":
            "Bilateral vs. absent"
    })
)

forest_multinomial["Estimate"] = (
    forest_multinomial["RRR"]
)

forest_multinomial["CI_lower"] = (
    forest_multinomial["RRR_CI95_lower"]
)

forest_multinomial["CI_upper"] = (
    forest_multinomial["RRR_CI95_upper"]
)

forest_multinomial["Adjusted_p"] = (
    forest_multinomial[
        "p_BH_axes_and_comparisons"
    ]
)

forest = pd.concat(
    [
        forest_gee,
        forest_multinomial
    ],
    ignore_index=True,
    sort=False
)

forest["Dimension"] = forest["Term"].replace({
    "Clinical_Dimension_1_z": "Dimension 1",
    "Clinical_Dimension_2_z": "Dimension 2"
})

forest_order = [
    (
        "GEE: side-specific arthralgia",
        "Present vs. absent",
        "Dimension 1"
    ),
    (
        "GEE: side-specific arthralgia",
        "Present vs. absent",
        "Dimension 2"
    ),
    (
        "Multinomial: patient-level pattern",
        "Unilateral vs. absent",
        "Dimension 1"
    ),
    (
        "Multinomial: patient-level pattern",
        "Unilateral vs. absent",
        "Dimension 2"
    ),
    (
        "Multinomial: patient-level pattern",
        "Bilateral vs. absent",
        "Dimension 1"
    ),
    (
        "Multinomial: patient-level pattern",
        "Bilateral vs. absent",
        "Dimension 2"
    )
]

forest["Plot_order"] = forest.apply(
    lambda row: forest_order.index((
        row["Model"],
        row["Comparison_label"],
        row["Dimension"]
    )),
    axis=1
)

forest = (
    forest
    .sort_values("Plot_order")
    .reset_index(drop=True)
)


# ============================================================
# 6. FIGURE 2
# CLINICAL DIMENSIONS AND ARTHRALGIA
# ============================================================

fig = plt.figure(
    figsize=(14.8, 8.2)
)

grid = fig.add_gridspec(
    nrows=2,
    ncols=2,
    width_ratios=[1, 1.18],
    height_ratios=[1, 1],
    wspace=0.42,
    hspace=0.46
)

ax_a = fig.add_subplot(
    grid[0, 0]
)

ax_b = fig.add_subplot(
    grid[1, 0]
)

ax_c = fig.add_subplot(
    grid[:, 1]
)

raincloud_plot(
    ax_a,
    analysis_data,
    variable="Clinical_Dimension_1",
    title="Dimension 1: pain–muscle burden",
    dimension_color=DIMENSION_1_COLOR
)

raincloud_plot(
    ax_b,
    analysis_data,
    variable="Clinical_Dimension_2",
    title="Dimension 2: crepitus-dominated pattern",
    dimension_color=DIMENSION_2_COLOR
)

add_panel_letter(ax_a, "A")
add_panel_letter(ax_b, "B")
add_panel_letter(ax_c, "C")

# Forest plot
y_positions = np.arange(
    len(forest)
)[::-1]

for position, (_, row) in zip(
    y_positions,
    forest.iterrows()
):

    color = (
        DIMENSION_1_COLOR
        if row["Dimension"] == "Dimension 1"
        else DIMENSION_2_COLOR
    )

    marker = (
        "o"
        if row["Model"] ==
        "GEE: side-specific arthralgia"
        else "s"
    )

    ax_c.errorbar(
        row["Estimate"],
        position,
        xerr=[[
            row["Estimate"] -
            row["CI_lower"]
        ], [
            row["CI_upper"] -
            row["Estimate"]
        ]],
        fmt=marker,
        markersize=8,
        color=color,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.9,
        ecolor=color,
        elinewidth=1.7,
        capsize=3.5,
        zorder=3
    )

    estimate_type = (
        "OR"
        if row["Model"] ==
        "GEE: side-specific arthralgia"
        else "RRR"
    )

    if row["Adjusted_p"] < 0.001:
        p_text = "adjusted p < 0.001"
    else:
        p_text = (
            f"adjusted p = "
            f"{row['Adjusted_p']:.3f}"
        )

    estimate_text = (
        f"{estimate_type} "
        f"{row['Estimate']:.2f} "
        f"({row['CI_lower']:.2f}–"
        f"{row['CI_upper']:.2f}); "
        f"{p_text}"
    )

    ax_c.text(
        3.12,
        position,
        estimate_text,
        va="center",
        fontsize=8.2,
        color=TEXT_COLOR
    )

y_labels = [
    f"{row['Comparison_label']}\n"
    f"{row['Dimension']}"
    for _, row in forest.iterrows()
]

ax_c.set_yticks(
    y_positions
)

ax_c.set_yticklabels(
    y_labels
)

ax_c.axvline(
    1,
    color="#5F6C78",
    linewidth=1.1,
    linestyle="--"
)

ax_c.set_xscale("log")

ax_c.set_xlim(
    0.45,
    4.9
)

ax_c.set_xticks([
    0.5,
    1,
    2,
    4
])

ax_c.get_xaxis().set_major_formatter(
    mpl.ticker.ScalarFormatter()
)

ax_c.set_xlabel(
    "Adjusted estimate per one-standard-deviation increase"
)

ax_c.set_title(
    "Adjusted association with arthralgia",
    loc="left",
    pad=12
)

# Visual separation between the models
ax_c.axhline(
    y=3.5,
    color="#CBD5DF",
    linewidth=1.0
)

ax_c.text(
    0.47,
    5.75,
    "Joint-level model — GEE",
    fontsize=8.5,
    fontweight="bold",
    color="#52606D"
)

ax_c.text(
    0.47,
    3.35,
    "Patient-level model — multinomial regression",
    fontsize=8.5,
    fontweight="bold",
    color="#52606D"
)

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=NEUTRAL_COLOR,
        markeredgecolor="white",
        markersize=8,
        label="GEE: OR"
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="none",
        markerfacecolor=NEUTRAL_COLOR,
        markeredgecolor="white",
        markersize=8,
        label="Multinomial: RRR"
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=DIMENSION_1_COLOR,
        markersize=8,
        label="Dimension 1"
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=DIMENSION_2_COLOR,
        markersize=8,
        label="Dimension 2"
    )
]

# Create a dedicated area below the comparisons for the legend
ax_c.set_ylim(-1.65, 6.25)

ax_c.legend(
    handles=legend_elements,
    frameon=False,
    ncol=4,
    loc="lower center",
    bbox_to_anchor=(0.50, 0.015),
    fontsize=8.1,
    handletextpad=0.45,
    columnspacing=1.15,
    borderaxespad=0
)




format_axis(ax_c)
ax_c.grid(axis="y", visible=False)

fig.suptitle(
    "Association between the clinical dimensions and arthralgia pattern",
    x=0.055,
    y=1.01,
    ha="left",
    fontsize=15,
    fontweight="bold",
    color="#17212B"
)

fig.text(
    0.055,
    -0.018,
    "Models were adjusted for age and sex. The GEE model was additionally "
    "adjusted for side and accounted for the correlation between the two "
    "temporomandibular joints of the same patient. OR: odds ratio; "
    "RRR: relative risk ratio.",
    ha="left",
    fontsize=8.5,
    color="#536170"
)

save_figure(
    fig,
    "Figure_2_clinical_dimensions_and_arthralgia"
)

plt.show()


# ============================================================
# 7. SUPPLEMENTARY FIGURE
# SENSITIVITY ANALYSIS EXCLUDING CREPITUS
# ============================================================

sensitivity_data = analysis_data.dropna(
    subset=[
        "Clinical_Dimension_1",
        "Sensitivity_Dimension_1_no_crepitus"
    ]
).copy()

rho, rho_p = spearmanr(
    sensitivity_data["Clinical_Dimension_1"],
    sensitivity_data[
        "Sensitivity_Dimension_1_no_crepitus"
    ]
)

sensitivity_result = sensitivity.iloc[0]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.8, 5.2),
    gridspec_kw={
        "width_ratios": [1.15, 0.85]
    }
)

# Panel A: agreement between scores
ax = axes[0]

sns.scatterplot(
    data=sensitivity_data,
    x="Clinical_Dimension_1",
    y="Sensitivity_Dimension_1_no_crepitus",
    hue="Arthralgia_pattern",
    hue_order=ARTHRALGIA_ORDER,
    palette=ARTHRALGIA_COLORS,
    s=45,
    alpha=0.68,
    edgecolor="white",
    linewidth=0.45,
    ax=ax
)

# Descriptive trend line
coefficients = np.polyfit(
    sensitivity_data["Clinical_Dimension_1"],
    sensitivity_data[
        "Sensitivity_Dimension_1_no_crepitus"
    ],
    deg=1
)

x_line = np.linspace(
    sensitivity_data[
        "Clinical_Dimension_1"
    ].min(),
    sensitivity_data[
        "Clinical_Dimension_1"
    ].max(),
    200
)

y_line = (
    coefficients[0] * x_line +
    coefficients[1]
)

ax.plot(
    x_line,
    y_line,
    color="#263746",
    linewidth=1.7,
    zorder=2
)

ax.text(
    0.04,
    0.94,
    f"Spearman ρ = {rho:.3f}",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10,
    fontweight="bold",
    bbox={
        "boxstyle": "round,pad=0.35",
        "facecolor": "white",
        "edgecolor": "#CFD8E1",
        "alpha": 0.92
    }
)

ax.set_xlabel(
    "Dimension 1 from the primary analysis"
)

ax.set_ylabel(
    "Dimension 1 after excluding crepitus"
)

ax.set_title(
    "Agreement between dimension scores",
    loc="left"
)

ax.legend(
    title="Arthralgia",
    frameon=False,
    fontsize=8.5,
    title_fontsize=9
)

format_axis(ax)
add_panel_letter(ax, "A")

# Panel B: comparison of adjusted estimates
ax = axes[1]

gee_dimension_1 = gee.loc[
    gee["Term"] ==
    "Clinical_Dimension_1_z"
].iloc[0]

sensitivity_estimates = pd.DataFrame({
    "Analysis": [
        "Primary analysis",
        "Without crepitus"
    ],
    "OR": [
        gee_dimension_1["OR"],
        sensitivity_result[
            "GEE_OR_per_SD"
        ]
    ],
    "Lower": [
        gee_dimension_1[
            "OR_CI95_lower"
        ],
        sensitivity_result[
            "GEE_OR_CI95_lower"
        ]
    ],
    "Upper": [
        gee_dimension_1[
            "OR_CI95_upper"
        ],
        sensitivity_result[
            "GEE_OR_CI95_upper"
        ]
    ]
})

sensitivity_y = [
    1,
    0
]

for y_position, (_, row) in zip(
    sensitivity_y,
    sensitivity_estimates.iterrows()
):

    ax.errorbar(
        row["OR"],
        y_position,
        xerr=[[
            row["OR"] - row["Lower"]
        ], [
            row["Upper"] - row["OR"]
        ]],
        fmt="o",
        markersize=9,
        color=DIMENSION_1_COLOR,
        markerfacecolor=DIMENSION_1_COLOR,
        markeredgecolor="white",
        markeredgewidth=1,
        ecolor=DIMENSION_1_COLOR,
        elinewidth=2,
        capsize=4
    )

    estimate_text = (
        f"OR {row['OR']:.2f} "
        f"({row['Lower']:.2f}–"
        f"{row['Upper']:.2f})"
    )

    ax.text(
        row["Upper"] + 0.06,
        y_position,
        estimate_text,
        va="center",
        fontsize=9
    )

ax.axvline(
    1,
    color="#5F6C78",
    linewidth=1.1,
    linestyle="--"
)

ax.set_yticks(
    sensitivity_y
)

ax.set_yticklabels(
    sensitivity_estimates["Analysis"]
)

ax.set_xlim(
    0.75,
    2.45
)

ax.set_xlabel(
    "Adjusted OR per one-standard-deviation increase"
)

ax.set_title(
    "Association with side-specific arthralgia",
    loc="left"
)

format_axis(ax)
ax.grid(axis="y", visible=False)
add_panel_letter(ax, "B")

fig.suptitle(
    "Sensitivity analysis after excluding crepitus",
    x=0.06,
    y=1.02,
    ha="left",
    fontsize=14,
    fontweight="bold",
    color="#17212B"
)

fig.text(
    0.06,
    -0.03,
    "The analysis excluding crepitus retained one clinical dimension. "
    "Estimates were obtained from GEE models adjusted for age, sex, and side.",
    ha="left",
    fontsize=8.5,
    color="#536170"
)

plt.tight_layout()

save_figure(
    fig,
    "Supplementary_Figure_S1_sensitivity_without_crepitus"
)

plt.show()


# ============================================================
# 8. LIST GENERATED FILES
# ============================================================

print("\nFigures successfully generated in:")
print(OUTPUT_FOLDER.resolve())

for generated_file in sorted(
    OUTPUT_FOLDER.iterdir()
):
    print(" -", generated_file.name)