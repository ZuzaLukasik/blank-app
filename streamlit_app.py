import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Model wzrostu i zysku", page_icon="📈", layout="wide")


def simulate(params):
	"""Run the model with market-clearing investment."""
	dt = params["dt"]
	steps = int(round(params["periods"] / dt))
	times = np.arange(steps + 1) * dt
	capital = np.zeros(len(times))
	labor = np.zeros(len(times))
	population = np.zeros(len(times))
	output = np.zeros(len(times))
	alpha = np.zeros(len(times))
	beta = np.zeros(len(times))
	alk = np.zeros(len(times))
	kor = np.zeros(len(times))
	klr = np.zeros(len(times))
	wage = np.zeros(len(times))
	consumption = np.zeros(len(times))
	investment = np.zeros(len(times))
	government = np.zeros(len(times))
	exports = np.zeros(len(times))
	imports = np.zeros(len(times))
	profit = np.zeros(len(times))
	demand = np.zeros(len(times))

	capital[0] = params["K0"]
	labor[0] = params["L0"]
	population[0] = params["Pop0"]
	alpha[0] = np.clip(params["alpha0"], 0.01, 0.99)
	employment_rate = params["L0"] / params["Pop0"]

	for t in range(steps + 1):
		beta[t] = 1 - alpha[t]
		population[t] = params["Pop0"] * np.exp(params["n"] * times[t])
		labor[t] = employment_rate * population[t]

		output[t] = (
			params["q0"] * params["E"] ** params["zeta"]
			* (capital[t] / params["K0"]) ** alpha[t]
			* (labor[t] / params["L0"]) ** beta[t]
		)
		alk[t] = (
			params["alk0"] * params["E"] ** (-params["theta"])
			* np.exp(-params["gamma"] * (klr[t - 1] / params["KLR0"] - 1))
			if t > 0 else params["alk0"]
		)
		kor[t] = capital[t] / max(output[t], 1e-12)
		wage[t] = beta[t] * output[t] / max(labor[t], 1e-12)
		klr[t] = alpha[t] * wage[t] / max((1 / alk[t] + params["R"]) * beta[t], 1e-12)

		consumption[t] = params["a"] * params["Pq"] * output[t]
		government[t] = params["GovSp"] * params["Pq"] * output[t]
		imports[t] = params["m0"] * params["Pq"] * output[t] * (params["alk0"] / max(alk[t], 1e-12)) ** params["eta"]
		exports[t] = params["x0"] * params["Pq"] * output[t] * (klr[t] / params["KLR0"]) ** params["kappa"]
		investment[t] = max(params["b"] * (consumption[t] - consumption[t - 1]), 0) if t > 0 else 0
		demand[t] = consumption[t] + investment[t] + government[t] + exports[t] - imports[t]


		profit[t] = output[t] * params["Pq"] - capital[t] * params["Pk"] * (1 / alk[t] + params["R"]) - labor[t] * wage[t] * params["Pq"]

		if t < steps:
			alpha[t + 1] = np.clip(
				kor[t] * (1 / max(alk[t], 1e-12) + params["R"]),
				0.02,
				0.98,
			)
			capital[t + 1] = max(capital[t] + dt * (investment[t] / params["Pq"] - capital[t] / max(alk[t], 1e-12)), 1e-8)

	output_growth = np.r_[np.nan, np.diff(output) / np.maximum(output[:-1], 1e-12) / dt]
	klr_growth = np.r_[np.nan, np.diff(klr) / np.maximum(klr[:-1], 1e-12) / dt]
	wage_growth = np.r_[np.nan, np.diff(wage) / np.maximum(wage[:-1], 1e-12) / dt]
	capital_growth = np.r_[np.nan, np.diff(capital) / np.maximum(capital[:-1], 1e-12) / dt]
	labor_growth = np.r_[np.nan, np.diff(labor) / np.maximum(labor[:-1], 1e-12) / dt]
	growth_identity = alpha * capital_growth + beta * labor_growth + kor * wage / np.maximum(klr, 1e-12) * np.log(np.maximum(klr / params["KLR0"], 1e-12)) * (klr_growth - wage_growth)

	return pd.DataFrame({
		"Okres": times, "Kapitał K": capital, "Praca L": labor, "Populacja": population,
		"Produkcja q": output, "Popyt Y": demand, "alpha": alpha,
		"beta": beta, "alk": alk, "KOR": kor, "KLR": klr, "Płaca rw": wage,
		"Konsumpcja C": consumption, "Inwestycje I": investment,
		"Wydatki G": government,
		"Eksport X": exports, "Import M": imports, "Zysk pi": profit,
		"Bezrobocie u": 1 - labor / population, "Luka popytowa": demand - params["Pq"] * output,
		"Wzrost q": output_growth, "Wzrost q z równania": growth_identity,
	})


def base_params(values):
	names = ["periods", "dt", "K0", "L0", "Pop0", "q0", "alpha0", "alk0", "E", "zeta", "theta", "gamma", "R", "n", "a", "b", "GovSp", "m0", "eta", "x0", "kappa", "Pq", "Pk"]
	params = {name: values[name] for name in names}
	params["KLR0"] = params["K0"] / params["L0"]
	return params


def scenario_inputs(label, key_prefix):
	st.subheader(label)
	periods = st.number_input("Horyzont symulacji", 5, 200, 40, key=f"{key_prefix}_periods")
	K0 = st.number_input("Kapitał K₀", 1.0, 1_000_000.0, 100.0, key=f"{key_prefix}_K0")
	L0 = st.number_input("Praca L₀", 1.0, 1_000_000.0, 100.0, key=f"{key_prefix}_L0")
	Pop0 = st.number_input("Populacja Pop₀", 1.0, 10_000_000.0, 120.0, key=f"{key_prefix}_Pop0")
	q0 = st.number_input("Produkcja bazowa q₀", 1.0, 1_000_000.0, 100.0, key=f"{key_prefix}_q0")
	alpha0 = st.slider("α₀", 0.02, 0.98, 0.35, 0.01, key=f"{key_prefix}_alpha0")
	alk0 = st.number_input("Średni okres użytkowania alk₀", 1.0, 100.0, 20.0, key=f"{key_prefix}_alk0")
	st.metric("Bazowe KLR₀ = K₀ / L₀", f"{K0 / L0:.4f}")
	st.subheader("Technologia i dynamika")
	if f"{key_prefix}_E" in st.session_state and not 0 < st.session_state[f"{key_prefix}_E"] <= 1:
		st.session_state[f"{key_prefix}_E"] = 1.0
	if f"{key_prefix}_zeta" in st.session_state and st.session_state[f"{key_prefix}_zeta"] <= 0:
		st.session_state[f"{key_prefix}_zeta"] = 0.10
	if f"{key_prefix}_theta" in st.session_state and st.session_state[f"{key_prefix}_theta"] <= 0:
		st.session_state[f"{key_prefix}_theta"] = 0.10
	E = st.slider("Czynnik środowiskowy E", 0.01, 1.0, 1.0, 0.01, key=f"{key_prefix}_E")
	zeta = st.number_input("ζ", 0.01, 5.0, 1.0, 0.1, key=f"{key_prefix}_zeta")
	theta = st.number_input("θ", 0.01, 5.0, 1.0, 0.1, key=f"{key_prefix}_theta")
	gamma = st.number_input("γ", -5.0, 5.0, 0.25, 0.05, key=f"{key_prefix}_gamma")
	R = st.slider("R", -0.05, 0.50, 0.05, 0.01, key=f"{key_prefix}_R")
	n = st.slider("Tempo wzrostu populacji n", -0.05, 0.10, 0.02, 0.005, key=f"{key_prefix}_n")
	st.subheader("Popyt i handel")
	a = st.slider("Skłonność do konsumpcji a", 0.0, 1.0, 0.70, 0.01, key=f"{key_prefix}_a")
	b = st.slider("Parametr inwestycji b", 0.0, 5.0, 1.00, 0.05, key=f"{key_prefix}_b")
	GovSp = st.slider("Wydatki rządowe / Y", 0.0, 0.5, 0.15, 0.01, key=f"{key_prefix}_GovSp")
	m0 = st.slider("Import / Y (m₀)", 0.0, 0.8, 0.20, 0.01, key=f"{key_prefix}_m0")
	eta = st.number_input("η", -5.0, 5.0, 0.5, 0.1, key=f"{key_prefix}_eta")
	x0 = st.slider("Eksport / Y (x₀)", 0.0, 0.8, 0.20, 0.01, key=f"{key_prefix}_x0")
	kappa = st.number_input("κ", -5.0, 5.0, 0.5, 0.1, key=f"{key_prefix}_kappa")
	Pq = st.number_input("Cena produktu Pq", 0.01, 100.0, 1.0, 0.1, key=f"{key_prefix}_Pq")
	Pk = st.number_input("Cena kapitału Pk", 0.01, 100.0, 1.0, 0.1, key=f"{key_prefix}_Pk")
	return {
		"periods": periods, "dt": 0.01, "K0": K0, "L0": L0, "Pop0": Pop0,
		"q0": q0, "alpha0": alpha0, "alk0": alk0, "E": E, "zeta": zeta,
		"theta": theta, "gamma": gamma, "R": R, "n": n, "a": a, "b": b,
		"GovSp": GovSp, "m0": m0, "eta": eta, "x0": x0, "kappa": kappa,
		"Pq": Pq, "Pk": Pk,
	}


st.title("Model wzrostu, produkcji i zysku")
st.caption("Dyskretna implementacja funkcji produkcji z opóźnioną adaptacją udziału kapitału.")

with st.sidebar:
	st.header("Parametry symulacji")
	st.caption("Krok obliczeń: dt = 0.01")
	st.session_state["scenario_a_E"] = 1.0
	st.session_state["scenario_a_zeta"] = 1.0
	st.session_state["scenario_a_theta"] = 1.0
	with st.expander("Scenariusz A", expanded=True):
		values = scenario_inputs("Parametry scenariusza A", "scenario_a")
	compare_scenarios = st.checkbox("Pokaż scenariusz B na wspólnych wykresach", value=True)
	if compare_scenarios:
		with st.expander("Scenariusz B", expanded=True):
			st.caption("Scenariusz B dziedziczy wszystkie parametry A poza E, ζ i θ.")
			values_b = dict(values)
			if "scenario_b_E" in st.session_state and not 0 < st.session_state["scenario_b_E"] < 1:
				st.session_state["scenario_b_E"] = 0.70
			values_b["E"] = st.slider("Czynnik środowiskowy E", 0.01, 0.99, 0.70, 0.01, key="scenario_b_E")
			values_b["zeta"] = st.number_input("ζ", 0.01, 5.0, 1.2, 0.1, key="scenario_b_zeta")
			values_b["theta"] = st.number_input("θ", 0.01, 5.0, 2.0, 0.1, key="scenario_b_theta")
		with st.expander("Scenariusz C", expanded=True):
			st.caption("C pokazuje korzystny wariant: niższe E i ujemne ζ zwiększają poziom produkcji.")
			values_c = dict(values)
			if "scenario_c_E" in st.session_state and not 0 < st.session_state["scenario_c_E"] < 1:
				st.session_state["scenario_c_E"] = 0.25
			values_c["E"] = st.slider("Czynnik środowiskowy E", 0.01, 0.99, 0.85, 0.01, key="scenario_c_E")
			if "scenario_c_zeta" in st.session_state and st.session_state["scenario_c_zeta"] <= 0:
				st.session_state["scenario_c_zeta"] = 2.0
			if "scenario_c_theta" in st.session_state and st.session_state["scenario_c_theta"] <= 0:
				st.session_state["scenario_c_theta"] = 2.5
			values_c["zeta"] = st.number_input("ζ", 0.01, 5.0, 2.0, 0.1, key="scenario_c_zeta")
			values_c["theta"] = st.number_input("θ", 0.01, 5.0, 2.5, 0.1, key="scenario_c_theta")
	else:
		values_b = None
		values_c = None

scenario_data = {"Scenariusz A": simulate(base_params(values))}
if values_b is not None:
	scenario_data["Scenariusz B"] = simulate(base_params(values_b))
if values_c is not None:
	scenario_data["Scenariusz C"] = simulate(base_params(values_c))
data = scenario_data["Scenariusz A"]


def scenario_chart(column, title=None):
	frames = []
	for scenario_name, frame in scenario_data.items():
		part = frame[["Okres", column]].copy()
		part["Scenariusz"] = scenario_name
		frames.append(part)
	chart_data = pd.concat(frames, ignore_index=True)
	return px.line(chart_data, x="Okres", y=column, color="Scenariusz", title=title or column)


def scenario_metric(formatter):
	return " | ".join(
		f"{scenario_name}: {formatter(frame)}"
		for scenario_name, frame in scenario_data.items()
	)


metric_cols = st.columns(5)
metric_cols[0].metric("Produkcja końcowa", scenario_metric(lambda frame: f"{frame['Produkcja q'].iloc[-1]:,.2f}"))
metric_cols[1].metric("Kapitał końcowy", scenario_metric(lambda frame: f"{frame['Kapitał K'].iloc[-1]:,.2f}"))
metric_cols[2].metric("α końcowe", scenario_metric(lambda frame: f"{frame['alpha'].iloc[-1]:.3f}"))
metric_cols[3].metric("Suma zysku", scenario_metric(lambda frame: f"{frame['Zysk pi'].sum():,.2f}"))
metric_cols[4].metric("Bezrobocie", scenario_metric(lambda frame: f"{100 * frame['Bezrobocie u'].iloc[-1]:.2f}%"))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Przebieg modelu", "Zysk i R", "Równowaga popytu", "Analiza wrażliwości", "Założenia"])
with tab1:
	st.subheader("Osobne wykresy wszystkich zmiennych")
	plot_columns = [column for column in data.columns if column != "Okres"]
	for index in range(0, len(plot_columns), 2):
		left, right = st.columns(2)
		with left:
			column = plot_columns[index]
			st.plotly_chart(scenario_chart(column), use_container_width=True, key=f"all_variables_left_{index}_{column}")
		if index + 1 < len(plot_columns):
			with right:
				column = plot_columns[index + 1]
				st.plotly_chart(scenario_chart(column), use_container_width=True, key=f"all_variables_right_{index}_{column}")
	st.dataframe(data.round(4), use_container_width=True, hide_index=True)

with tab2:
	st.plotly_chart(scenario_chart("Zysk pi", "Zysk w czasie"), use_container_width=True, key="profit_chart")
	st.markdown("#### Czy `R` powinno zmienić funkcję celu?")
	st.write("Nie. Dla ustalonego R funkcja celu pozostaje: π = q·Pq − K·Pk·(1/alk + R) − L·rw·Pq. R staje się zmienną decyzyjną dopiero wtedy, gdy chcemy dobrać jego wartość maksymalizującą zysk.")
	if st.button("Znajdź R maksymalizujące sumę zysku"):
		candidates = np.linspace(-0.05, 0.50, 111)
		scores = []
		for candidate in candidates:
			candidate_values = dict(values)
			candidate_values["R"] = float(candidate)
			scores.append(simulate(base_params(candidate_values))["Zysk pi"].sum())
		best = int(np.argmax(scores))
		st.success(f"Najlepsze R w siatce: {candidates[best]:.3f}; suma zysku: {scores[best]:,.2f}")
		st.plotly_chart(px.line(x=candidates, y=scores, labels={"x": "R", "y": "Suma zysku"}, title="Funkcja celu względem R"), use_container_width=True, key="r_optimization_chart")

with tab3:
	accounting_gap = (data["Produkcja q"] * values["Pq"] - data["Popyt Y"]).abs().max()
	if accounting_gap < 1e-10:
		st.success("Równowaga rynkowa spełniona: produkcja = popyt w każdym okresie.")
	else:
		st.warning(f"Równowaga rynkowa niespełniona. Luka: {accounting_gap:.2e}")
	st.write("Inwestycje I wynikają z reguły behawioralnej I = b · ΔC. Parametry popytu są ustawiane ręcznie osobno dla każdego scenariusza.")
	st.plotly_chart(scenario_chart("Inwestycje I", "Inwestycje I"), use_container_width=True, key="investment_chart")
	st.plotly_chart(scenario_chart("Wzrost q", "Wzrost q"), use_container_width=True, key="output_growth_chart")
	st.plotly_chart(scenario_chart("Wzrost q z równania", "Wzrost q z równania"), use_container_width=True, key="output_growth_identity_chart")

with tab4:
	st.subheader("Analiza wrażliwości")
	st.write("Każdy wykres zmienia jeden parametr względem scenariusza A, a pozostałe parametry pozostają stałe.")
	st.info("Wariant bazowy: A = E 1.0, ζ 1.0, θ 1.0. Warianty B i C pokazują odchylenia od tej bazy.")

	sensitivity_specs = [
		("E", [0.10, 0.25, 0.40, 0.70, 0.85, 1.0], "Czynnik środowiskowy E"),
		("zeta", [0.1, 0.5, 1.0, 1.5, 2.0, 2.5], "Parametr ζ"),
		("theta", [0.1, 0.5, 1.0, 1.5, 2.0, 2.5], "Parametr θ"),
	]

	comparison_rows = []
	for scenario_name, frame in scenario_data.items():
		comparison_rows.append({
			"Scenariusz": scenario_name,
			"E": values["E"] if scenario_name == "Scenariusz A" else (values_b if scenario_name == "Scenariusz B" else values_c)["E"],
			"ζ": values["zeta"] if scenario_name == "Scenariusz A" else (values_b if scenario_name == "Scenariusz B" else values_c)["zeta"],
			"θ": values["theta"] if scenario_name == "Scenariusz A" else (values_b if scenario_name == "Scenariusz B" else values_c)["theta"],
			"Produkcja końcowa": frame["Produkcja q"].iloc[-1],
			"Kapitał końcowy": frame["Kapitał K"].iloc[-1],
			"Suma zysku": frame["Zysk pi"].sum(),
			"Bezrobocie końcowe": 100 * frame["Bezrobocie u"].iloc[-1],
		})
	st.markdown("#### Bezpośrednie porównanie scenariuszy")
	st.dataframe(pd.DataFrame(comparison_rows).round(3), use_container_width=True, hide_index=True)

	for parameter, candidates, title in sensitivity_specs:
		rows = []
		for candidate in candidates:
			candidate_values = dict(values)
			candidate_values[parameter] = candidate
			candidate_data = simulate(base_params(candidate_values))
			rows.append({
				parameter: candidate,
				"Produkcja końcowa": candidate_data["Produkcja q"].iloc[-1],
				"Kapitał końcowy": candidate_data["Kapitał K"].iloc[-1],
				"Suma zysku": candidate_data["Zysk pi"].sum(),
				"Bezrobocie końcowe": 100 * candidate_data["Bezrobocie u"].iloc[-1],
			})
		sensitivity_data = pd.DataFrame(rows)
		st.markdown(f"#### Wrażliwość na {title}")
		left, right = st.columns(2)
		with left:
			st.plotly_chart(
				px.line(sensitivity_data, x=parameter, y=["Produkcja końcowa", "Kapitał końcowy"], markers=True, title="Produkcja i kapitał"),
				use_container_width=True,
				key=f"sensitivity_levels_{parameter}",
			)
		with right:
			st.plotly_chart(
				px.line(sensitivity_data, x=parameter, y=["Suma zysku", "Bezrobocie końcowe"], markers=True, title="Zysk i bezrobocie"),
				use_container_width=True,
				key=f"sensitivity_results_{parameter}",
			)
		st.dataframe(sensitivity_data.round(3), use_container_width=True, hide_index=True)

with tab5:
	st.markdown("""
Model aktualizuje udział kapitału regułą **αₙ₊₁ = KORₙ · (1/alkₙ + R)**. Wartość α jest ograniczana do przedziału (0, 1) wyłącznie dla stabilności numerycznej.

	- `alk₀` jest wartością początkową i zawsze zachodzi `alk(0) = alk₀`, niezależnie od `E` i `θ`.
	- Stałe E i ζ wpływają na poziom q, ale nie dodają bezpośredniego składnika do stopy wzrostu.
	- `alk` wykorzystuje KLR z poprzedniego kroku, co zapobiega sprzężeniu algebraicznemu.
	- Inwestycje `I = max(b · ΔC, 0)` są wyznaczane behawioralnie; kalibracja parametrów popytu minimalizuje lukę `Pq · q − (C + I + G + X − M)`, a kapitał przechodzi dalej zgodnie z `Kₙ₊₁ = Kₙ + dt · (Iₙ/Pq − Kₙ/alkₙ)`.
- Przy `Pq = Pk = 1` funkcja zysku upraszcza się dokładnie do postaci podanej w opisie.
""")
