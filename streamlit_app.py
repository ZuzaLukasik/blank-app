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
	planned_demand = np.zeros(len(times))
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
		kor[t] = capital[t] / max(output[t], 1e-12)
		wage[t] = beta[t] * output[t] / max(labor[t], 1e-12)
		if t == 0:
			alk[t] = params["alk0"]
		else:
			alk[t] = alk[t - 1]
			for _ in range(100):
				klr[t] = alpha[t] * wage[t] / max((1 / alk[t] + params["R"]) * beta[t], 1e-12)
				updated_alk = params["alk0"] * np.exp(
					(
						-params["gamma0"]
						- params["gamma_E"] * (1 - params["E"])
					)
					* (klr[t] / params["KLR0"] - 1)
				)
				if np.isclose(updated_alk, alk[t], rtol=1e-10, atol=1e-12):
					alk[t] = updated_alk
					break
				alk[t] = updated_alk
		klr[t] = alpha[t] * wage[t] / max((1 / alk[t] + params["R"]) * beta[t], 1e-12)

		consumption[t] = params["a"] * params["Pq"] * output[t]
		government[t] = params["GovSp"] * params["Pq"] * output[t]
		imports[t] = params["m0"] * params["Pq"] * output[t] * (params["alk0"] / max(alk[t], 1e-12)) ** params["eta"]
		exports[t] = params["x0"] * params["Pq"] * output[t] * (klr[t] / params["KLR0"]) ** params["kappa"]
		planned_investment = max(params["b"] * (consumption[t] - consumption[t - 1]), 0) if t > 0 else 0
		planned_demand[t] = consumption[t] + planned_investment + government[t] + exports[t] - imports[t]
		investment[t] = planned_investment + params["Pq"] * output[t] - planned_demand[t]
		demand[t] = consumption[t] + investment[t] + government[t] + exports[t] - imports[t]


		profit[t] = output[t] * params["Pq"] - capital[t] * params["Pk"] * (1 / alk[t] + params["R"]) - labor[t] * wage[t] * params["Pq"]

		if t < steps:
			alpha[t + 1] = np.clip(
				kor[t] * (1 / max(alk[t], 1e-12) + params["R"]),
				0.02,
				0.98,
			)
			capital[t + 1] = max(capital[t] + dt * (investment[t] / params["Pq"] - capital[t] / max(alk[t], 1e-12)), 1e-8)

	klr_growth = np.r_[np.nan, np.diff(klr) / np.maximum(klr[:-1], 1e-12) / dt]
	wage_growth = np.r_[np.nan, np.diff(wage) / np.maximum(wage[:-1], 1e-12) / dt]
	capital_growth = np.r_[np.nan, np.diff(capital) / np.maximum(capital[:-1], 1e-12) / dt]
	labor_growth = np.r_[np.nan, np.diff(labor) / np.maximum(labor[:-1], 1e-12) / dt]
	growth_identity = (
		alpha * capital_growth
		+ beta * labor_growth
		+ kor * wage / np.maximum(klr, 1e-12)
		* np.log(np.maximum(klr / params["KLR0"], 1e-12))
		* (klr_growth - wage_growth)
	)

	return pd.DataFrame({
		"Okres": times, "Kapitał K": capital, "Praca L": labor, "Populacja": population,
		"Produkcja q": output, "Popyt planowany Yp": planned_demand, "Popyt Y": demand, "alpha": alpha,
		"beta": beta, "alk": alk, "KOR": kor, "KLR": klr, "Płaca rw": wage,
		"Konsumpcja C": consumption, "Inwestycje I": investment,
		"Wydatki G": government,
		"Eksport X": exports, "Import M": imports, "Zysk pi": profit,
		"Bezrobocie u": 1 - labor / population, "Luka popytowa": demand - params["Pq"] * output,
		"Gq": growth_identity,
	})


def base_params(values):
	names = ["periods", "dt", "K0", "L0", "Pop0", "q0", "alk0", "E", "zeta", "gamma", "gamma_E", "gamma0", "R", "n", "a", "b", "GovSp", "m0", "eta", "x0", "kappa", "Pq", "Pk"]
	params = {name: values[name] for name in names}
	params["KLR0"] = params["K0"] / params["L0"]
	params["alpha0"] = np.clip(
		(params["K0"] / params["q0"]) * (1 / params["alk0"] + params["R"]),
		0.02,
		0.98,
	)
	return params


def scenario_inputs(label, key_prefix):
	st.subheader(label)
	periods = st.number_input("Horyzont symulacji", 5, 200, 40, key=f"{key_prefix}_periods")
	K0 = st.number_input("Kapitał K₀", 1.0, 1_000_000.0, 100.0, key=f"{key_prefix}_K0")
	L0 = st.number_input("Praca L₀", 1.0, 1_000_000.0, 100.0, key=f"{key_prefix}_L0")
	Pop0 = st.number_input("Populacja Pop₀", 1.0, 10_000_000.0, 108.0, key=f"{key_prefix}_Pop0_v2")
	q0 = st.number_input("Produkcja bazowa q₀", 1.0, 1_000_000.0, 100.0, key=f"{key_prefix}_q0")
	alk0 = st.number_input("Średni okres użytkowania alk₀", 1.0, 100.0, 10.0, key=f"{key_prefix}_alk0_v2")
	st.metric("Bazowe KLR₀ = K₀ / L₀", f"{K0 / L0:.4f}")
	st.subheader("Technologia i dynamika")
	if f"{key_prefix}_E" in st.session_state and not 0 < st.session_state[f"{key_prefix}_E"] <= 1:
		st.session_state[f"{key_prefix}_E"] = 1.0
	if f"{key_prefix}_zeta" in st.session_state and st.session_state[f"{key_prefix}_zeta"] <= 0:
		st.session_state[f"{key_prefix}_zeta"] = 0.10
	E = st.slider("Czynnik środowiskowy E", 0.01, 1.0, 1.0, 0.01, key=f"{key_prefix}_E")
	zeta = st.number_input("ζ", 0.01, 5.0, 1.0, 0.1, key=f"{key_prefix}_zeta")
	gamma = st.number_input("γ", -5.0, 5.0, 0.25, 0.05, key=f"{key_prefix}_gamma")
	gamma_E = st.number_input("γ_E", -5.0, 5.0, 0.25, 0.05, key=f"{key_prefix}_gamma_E")
	gamma0 = st.number_input("γ₀", -5.0, 5.0, 0.25, 0.05, key=f"{key_prefix}_gamma0")
	R = st.slider("R", -0.05, 0.50, 0.05, 0.01, key=f"{key_prefix}_R")
	alpha0 = np.clip((K0 / q0) * (1 / alk0 + R), 0.02, 0.98)
	st.caption(f"α₀ jest wyliczane ze wzoru i nie można go ustawić ręcznie: α₀ = {alpha0:.4f}")
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
		"q0": q0, "alk0": alk0, "E": E, "zeta": zeta,
		"gamma": gamma, "R": R, "n": n, "a": a, "b": b, "gamma_E": gamma_E, "gamma0": gamma0,
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
	with st.expander("Scenariusz A", expanded=True):
		values = scenario_inputs("Parametry scenariusza A", "scenario_a")
	compare_scenarios = st.checkbox("Pokaż scenariusz B na wspólnych wykresach", value=True)
	if compare_scenarios:
		with st.expander("Scenariusz B", expanded=True):
			st.caption("Scenariusz B dziedziczy parametry A z możliwością zmiany E, ζ, γ₀ i γ_E.")
			values_b = dict(values)
			if "scenario_b_E" in st.session_state and not 0 < st.session_state["scenario_b_E"] < 1:
				st.session_state["scenario_b_E"] = 0.70
			values_b["E"] = st.slider("Czynnik środowiskowy E", 0.01, 0.99, 0.70, 0.01, key="scenario_b_E")
			values_b["zeta"] = st.number_input("ζ", 0.01, 5.0, 1.2, 0.1, key="scenario_b_zeta")
			values_b["gamma0"] = st.number_input("γ₀", -5.0, 5.0, 0.25, 0.05, key="scenario_b_gamma0")
			values_b["gamma_E"] = st.number_input("γ_E", -5.0, 5.0, 0.25, 0.05, key="scenario_b_gamma_E")
		with st.expander("Scenariusz C", expanded=True):
			st.caption("C pokazuje korzystny wariant: niższe E i ujemne ζ zwiększają poziom produkcji.")
			values_c = dict(values)
			if "scenario_c_E" in st.session_state and not 0 < st.session_state["scenario_c_E"] < 1:
				st.session_state["scenario_c_E"] = 0.25
			values_c["E"] = st.slider("Czynnik środowiskowy E", 0.01, 0.99, 0.85, 0.01, key="scenario_c_E")
			if "scenario_c_zeta" in st.session_state and st.session_state["scenario_c_zeta"] <= 0:
				st.session_state["scenario_c_zeta"] = 2.0
			values_c["zeta"] = st.number_input("ζ", 0.01, 5.0, 2.0, 0.1, key="scenario_c_zeta")
			values_c["gamma0"] = st.number_input("γ₀", -5.0, 5.0, 0.25, 0.05, key="scenario_c_gamma0")
			values_c["gamma_E"] = st.number_input("γ_E", -5.0, 5.0, 0.25, 0.05, key="scenario_c_gamma_E")
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Przebieg modelu", "Zysk i R", "Równowaga popytu", "Analiza wrażliwości", "Założenia", "Wzory"])
with tab1:
	st.subheader("Osobne wykresy wszystkich zmiennych")
	st.plotly_chart(scenario_chart("alk", "Średni okres użytkowania kapitału alk"), use_container_width=True, key="alk_chart")
	st.plotly_chart(scenario_chart("KLR", "Relacja kapitał-praca KLR"), use_container_width=True, key="klr_chart")
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
	st.write("Popyt planowany wykorzystuje behawioralną inwestycję Iᵖ = b · ΔC. Inwestycja faktyczna jest korygowana o różnicę między produkcją a popytem planowanym, aby zapewnić równowagę rynkową.")
	st.plotly_chart(scenario_chart("Inwestycje I", "Inwestycje I"), use_container_width=True, key="investment_chart")
	st.plotly_chart(scenario_chart("Gq", "Gq z równania"), use_container_width=True, key="output_growth_chart")

with tab4:
	st.subheader("Analiza wrażliwości")
	st.write("Każdy wykres zmienia jeden parametr względem scenariusza A, a pozostałe parametry pozostają stałe.")
	st.info("Wariant bazowy: A = E 1.0, ζ 1.0, γ₀ 0.25, γ_E 0.25. Warianty B i C pokazują odchylenia od tej bazy.")

	sensitivity_specs = [
		("E", [0.10, 0.25, 0.40, 0.70, 0.85, 1.0], "Czynnik środowiskowy E"),
		("zeta", [0.1, 0.5, 1.0, 1.5, 2.0, 2.5], "Parametr ζ"),
		("gamma0", [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0], "Parametr γ₀"),
		("gamma_E", [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0], "Parametr γ_E"),
	]

	comparison_rows = []
	for scenario_name, frame in scenario_data.items():
		comparison_rows.append({
			"Scenariusz": scenario_name,
			"E": values["E"] if scenario_name == "Scenariusz A" else (values_b if scenario_name == "Scenariusz B" else values_c)["E"],
			"ζ": values["zeta"] if scenario_name == "Scenariusz A" else (values_b if scenario_name == "Scenariusz B" else values_c)["zeta"],
			"γ₀": values["gamma0"] if scenario_name == "Scenariusz A" else (values_b if scenario_name == "Scenariusz B" else values_c)["gamma0"],
			"γ_E": values["gamma_E"] if scenario_name == "Scenariusz A" else (values_b if scenario_name == "Scenariusz B" else values_c)["gamma_E"],
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
Model wyznacza początkowy udział kapitału ze wzoru **α₀ = KOR₀ · (1/alk₀ + R)**, a następnie aktualizuje go regułą **αₙ₊₁ = KORₙ · (1/alkₙ + R)**. Wartość α jest ograniczana do przedziału (0, 1) wyłącznie dla stabilności numerycznej.

	- `alk₀` jest wartością początkową i zawsze zachodzi `alk(0) = alk₀`, niezależnie od `E`.
	- Stałe E i ζ wpływają na poziom q, ale nie dodają bezpośredniego składnika do stopy wzrostu.
	- `alk` wykorzystuje KLR z poprzedniego kroku, co zapobiega sprzężeniu algebraicznemu.
	- Popyt planowany opiera się na `Iᵖ = max(b · ΔC, 0)`, a inwestycja faktyczna jest korygowana tak, aby `Pq · q = C + I + G + X − M`; kapitał przechodzi dalej zgodnie z `Kₙ₊₁ = Kₙ + dt · (Iₙ/Pq − Kₙ/alkₙ)`.
- Przy `Pq = Pk = 1` funkcja zysku upraszcza się dokładnie do postaci podanej w opisie.
""")

with tab6:
	st.subheader("Wzory używane w modelu")
	st.caption("Równania są zapisywane dla okresu n. Zmienne z indeksem n−1 pochodzą z poprzedniego kroku symulacji.")

	st.markdown("#### Warunki początkowe i parametry")
	st.latex(r"KLR_0 = \frac{K_0}{L_0}")
	st.latex(r"\beta_n = 1 - \alpha_n")
	st.latex(r"Pop_n = Pop_0 e^{n \cdot t_n}")
	st.latex(r"L_n = \frac{L_0}{Pop_0} \cdot Pop_n")

	st.markdown("#### Produkcja i technologia")
	st.latex(r"q_n = q_0 \cdot E^{\zeta} \cdot \left(\frac{K_n}{K_0}\right)^{\alpha_n} \cdot \left(\frac{L_n}{L_0}\right)^{\beta_n}")
	st.latex(r"alk_n = alk_0 \cdot e^{[\gamma_0 - \gamma_E(1-E)]\left(\frac{KLR_n}{KLR_0} - 1\right)} \quad (n > 0)")
	st.latex(r"alk_0 = alk_0")
	st.latex(r"KOR_n = \frac{K_n}{q_n}")
	st.latex(r"rw_n = \frac{\beta_n q_n}{L_n}")
	st.latex(r"KLR_n = \frac{\alpha_n rw_n}{\left(\frac{1}{alk_n} + R\right)\beta_n}")
	st.latex(r"\alpha_0 = \operatorname{clip}\left[KOR_0\left(\frac{1}{alk_0} + R\right),\ 0.02,\ 0.98\right]")
	st.latex(r"\alpha_{n+1} = \operatorname{clip}\left[KOR_n\left(\frac{1}{alk_n} + R\right),\ 0.02,\ 0.98\right]")

	st.markdown("#### Popyt, handel i równowaga rynkowa")
	st.latex(r"C_n = a \cdot P_q \cdot q_n")
	st.latex(r"G_n = GovSp \cdot P_q \cdot q_n")
	st.latex(r"M_n = m_0 \cdot P_q \cdot q_n \cdot \left(\frac{alk_0}{alk_n}\right)^{\eta}")
	st.latex(r"X_n = x_0 \cdot P_q \cdot q_n \cdot \left(\frac{KLR_n}{KLR_0}\right)^{\kappa}")
	st.latex(r"I_n^p = \max\left[b(C_n - C_{n-1}),\ 0\right], \quad I_0^p = 0")
	st.latex(r"Y_n^p = C_n + I_n^p + G_n + X_n - M_n")
	st.latex(r"I_n = I_n^p + P_q q_n - Y_n^p")
	st.latex(r"Y_n = C_n + I_n + G_n + X_n - M_n")
	st.latex(r"P_q q_n = Y_n")

	st.markdown("#### Kapitał, zysk i wskaźniki")
	st.latex(r"K_{n+1} = \max\left[K_n + \Delta t\left(\frac{I_n}{P_q} - \frac{K_n}{alk_n}\right),\ 10^{-8}\right]")
	st.latex(r"\pi_n = P_q q_n - P_k K_n\left(\frac{1}{alk_n} + R\right) - P_q L_n rw_n")
	st.latex(r"u_n = 1 - \frac{L_n}{Pop_n}")
	st.latex(r"g_{x,n} = \frac{x_n - x_{n-1}}{x_{n-1}\Delta t}")
	st.latex(r"g_{q,n}^{\mathrm{równanie}} = \alpha_n g_{K,n} + \beta_n g_{L,n} + KOR_n \frac{rw_n}{KLR_n} \ln\left(\frac{KLR_n}{KLR_0}\right) \left[\frac{\dot{KLR}_n}{KLR_n} - \frac{\dot{rw}_n}{rw_n}\right]")
