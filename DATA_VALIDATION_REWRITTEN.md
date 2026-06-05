2.2 Patient Baseline Data Validation
The goal for the data generation algorithm was to generate data that can be used to demonstrate multivariate time-series input for LSTMs in medical context. The code generates time series data, with logical correlations between the input parameters at a level of abstraction that is sufficient to demonstrate the level of complexity.

The previously mentioned rules were taken from educational material and later backed by medical research. Most of the rules are strongly supported (high confidence), implying there exists undeniable evidence in research to establish its truth. Some (rules) are moderately supported, i.e. few research have backed the claims. Few are weakly supported, implying there exists mixed views and results in scientific studies. One rule was found to be strongly contradicted (absolute bullshit) by multiple studies and is pointed out (see **).

Age > 80 | Systolic BP +15 mmHG
Strongly supported.
Direct Evidence: Study explicitly documents age-stratified BP changes, showing SBP marked increase (↑ ↑) in >80 age group. Framingham Study data: “lifetime risk of developing hypertension is > 90% for individuals aged 55-65.” Mechanism: arterial stiffening from loss of elastin, increased collagen/fibrosis. [Benetos, A., et al. (2019). Hypertension Management in Older and Frail Older Patients. Circulation Research, 124(7), 1008-1033. https://doi.org/10.1161/CIRCRESAHA.118.313236.]
Age >60 | Systolic BP +10 mmHg
Strongly supported.
Direct Evidence: Table 1 shows age 65-80 group has marked SBP elevation (↑) compared to younger ages. Progressive increase with age is well-established. [Benetos, A., et al. (2019). Hypertension Management in Older and Frail Older Patients. Circulation Research, 124(7), 1008-1033. https://doi.org/10.1161/CIRCRESAHA.118.313236.]
Table 1. Schematic Representation of the Various BP Profiles in Older Subjects
Age, y
SBP
DBP
BP Regulation Physio- -patholohy
Main Risks
Better BP Risk Marker
Management
65-80
↑ ↑
↑
High PR and AS
CV complications, cognitive decline
High SBP
Physical activities, assess TOD and global CVR, medical tt (SBP <140)
65-80
↑
↔ ↓
High AS
CV complications, cognitive decline
High SBP, PP, low DBP
Physical activities, assess TOD and global CVR, medical tt (SBP <140)
>80
↑ ↑
↔ ↓
High As
CV complications, falls
High PP, low DBP, OH
CGA, medical tt (SBP <150 or SBP <140 according functional status)
>80
↔ ↓
↔ ↓
High AS and comorbidities
CV complications, falls, loss of autonomy
Normal/low SBP, low DBP; normal/high PP, OH
CGA, deprescribing if SBP < 130 or OH, fight polypharmacy


AS indicates arterial stiffness; BP, blood pressure; CGA, Comprehensive Geriatric Assessment; CV, cardiovascular; CVR, cardiovascular risk; DBP, diastolic blood pressure; OH, orthostatic hypertension; PP, pulse pressure; PR, peripheral resistance; SBP, systolic blood pressure; TOD, target organ damage; and tt, treatment.
Female Gender | Heart Rate +5 bpm
Strongly supported.
Direct Evidence: Study suggests smaller female heart “needs to beat at a faster rate” and the women have “different intrinsic rhythmicity" of the pacemaker, causing faster beating. [Regitz-Zagrosek, V., & Kararigas, G. (2014). Role of Biological Sex in Normal Cardiac Function and in Its Response to Disease. Journal of Clinical and Diagnostic Research, 8(8), BE01-BE04. https://doi.org/10.7860/JCDR/2014/9635.4771.]
Active Lifestyle | Heart Rate -10 bpm
Strongly supported.
Direct Evidence: “Endurance trained athletes are well known to have low resting heart rates, with values below 30 beats per minute reported. Such low resting heart rates result from long periods of endurance training” due to increased parasympathetic (vagal) tone. [Coote, J. H., & Danson, E. J. (2015). Bradycardia in the trained athletes is attributable to high vagal tone. European Journal of Applied Physiology, 116(4), 701-708. https://doi.org/10.1113/jphysiol.2014.284364.]
Sedentary Lifestyle | Heart Rate +5 bpm
Strongly supported.
Hypothesis Tested: “Higher time spent in sedentary behavior would be associated with higher resting HR and lower resting overall variability, indicating cardiac-autonomic dysregulation in adults.” [Abdullah, B., A., et al. (2021). Associations of Sedentary Time with Heart Rate and Heart Rate Variability. PubMed Central, 18(16), 8508. https://doi.org/10.3390/ijerph18168508.]
Hypertension | SBP +20, DBP +10 mmHq
Strongly supported.
Direct Evidence: “Significant gender differences in both systolic pressure (p=0.003) with mean difference = 18.08 mmHg (CI: 16.13-19.9) and diastolic pressure (p=0.011) with mean difference = 3.6 mmHg (Cl: 2.06-5.14), higher in males than females. ”Study context: Pre-hypertensive (130-139/80-89) vs Normal (<120/80). Your +20 SBP adjustment aligns with difference between normal (120) and Stage 1 hypertension (140). [Oumer, A., et al. (2018). Blood Pressure and its Association with Gender. Body Mass Index, and Other Demographic Characteristics in a Large University Population. PLoS ONE, 13(5), e0195621. https://doi.org/10.1155/2018/4186496.]
Type 2 Diabetes | Blood Glucose 150 mg/dL
Strongly supported.
Direct Evidence: “The optimal HbA1c range for T2D is 7.1-7.7% regardless of diabetes duration.” Conversion: HbA1c 7.0-7.7% corresponds to average plasma glucose 126-153 mg/dL. Meta-analysis of 15 RCTs; your 150 mg/dL baseline falls perfectly within this evidence-based target range. [Basson, S., et al. (2021). A Target HbA1c Between 7 and 7.7% Reduces Microvascular and Macrovascular Complications in Type 2 Diabetes. Frontiers in Endocrinology, 12(6), 635251. https://doi.org/10.1007/s13300-021-01062-6.] 
Obesity | SBP +10, DBP +5, Heart Rate +5 bpm
Strongly supported.
Direct Evidence: “Visceral or central obesity…adipose tissue secretes adipocytokines… an important effect of adipocytokines is the production of arterial hypertension. Visceral obesity is the leading cause of MetS, which explains how it links to hypertension.” Mechanism: “Obesity and insulin resistance promotes the development of hypertension through multiple pathophysiological pathways.” [Arcone, R., & Sankar, B. (2023). Links between Metabolic Syndrome and Hypertension. Nutrients, 13(1), 87. https://doi.org/10.3390/metabo13010087.]
Chronic Kidney Disease | SBP +15, DBP +5 mmHg
Strongly supported.
Direct Evidence: “Hypertension is an important cause of chronic kidney disease (CKD). SBP was associated with incident CKD, with a steady increase in risk of incident CKD above an SBP of 120 mmHg.” [Go, A. S., et al. (2010). Relationship between Blood Pressure and Indecent Chronic Kidney Disease in Patients with Hypertension. JASN, 21(12), 2099-2110. https://doi.org/10.2215/CJN.02240311.]
Direct Evidence: “Hypertension affects the great majority of patients with chronic kidney disease (CKD). Both are intrinsically related; Hypertension is a strong determinant of worse renal and cardiovascular outcomes.” [Burnier, M., et al. (2023). Hypertension as Cardiovascular Risk Factor in Chronic Kidney Disease. Circulation Research, 132(7), 765-788. https://doi.org/10.1161/CIRCRESAHA.122.321762.]
CKD causes secondary hypertension via fluid retention + RAAS activation; your +15 mmHg SBP is well-supported.
COPD | SpO2 -4%, Respiratory Rate +4 breaths/min
Strongly supported.
Direct Evidence (SpO2): “Alveolar hypoxia and consequent hypoxemia increase in prevalence as COPD severity increases. The principal contributor to hypoxemia in COPD patients is ventilation/perfusion (V/Q) mismatch… COPD patients commonly present with SpO2 88-94% (normal 95-100%).” Direct Evidence (RR): “COPD causes increased neural drive to the respiratory muscles as the disease progresses… patients show… compensatory tachypnea… to maintain gas exchange.” [Wirth, J. M., & Coote, J. H. (2011). Sleep, COPD, and Hypoxemia. Sleep Medicine Reviews, 15(2), 81-87. https://doi.org/10.2147/COPD.S10611.]
Atrial Fibrillation | Heart Rate +15 bpm
Strongly supported.
Direct Evidence: “AF (Atrial Fibrillation) is a chaotic, rapid (300–500 bpm), and irregular atrial rhythm.” [Joglar JA, Chung MK, Armbruster AL, Benjamin EJ, Chyou JY, Cronin EM, Deswal A, Eckhardt LL, Goldberger ZD, Gopinathannair R, Gorenek B, Hess PL, Hlatky M, Hogan G, Ibeh C, Indik JH, Kido K, Kusumoto F, Link MS, Linta KT, Marcus GM, McCarthy PM, Patel N, Patton KK, Perez MV, Piccini JP, Russo AM, Sanders P, Streur MM, Thomas KL, Times S, Tisdale JE, Valente AM, Van Wagoner DR; Peer Review Committee Members. 2023. ACC/AHA/ACCP/HRS. Guideline for the Diagnosis and Management of Atrial Fibrillation: A Report of the American College of Cardiology/American Heart Association Joint Committee on Clinical Practice Guidelines. Circulation. 2024 Jan 2;149(1):e1-e156. doi: 10.1161/CIR.0000000000001193. Epub 2023 Nov 30. Erratum in: Circulation. 2024 Jan 2;149(1):e167. doi: 10.1161/CIR.0000000000001207. Erratum in: Circulation. 2024 Feb 27;149(9):e936. doi: 10.1161/CIR.0000000000001218. Erratum in: Circulation. 2024 Jun 11;149(24):e1413. doi: 10.1161/CIR.0000000000001263. PMID: 38033089; PMCID: PMC11095842. https://doi.org/10.1161/CIR.0000000000001193.]
+15 bpm is conservative. Patients show 20-40+ bpm elevation at baseline.
Age >80 | Heart Rate -5 bpm
Moderately supported.
Direct Evidence: “During rest, the older heart functions in almost the same way as a younger heart, except the heart rate (number of times the heart beats within a minute) is slightly lower.” [Richard G. Stefanacci (2024). Physical Changes With Aging. MSD Manual. https://www.msdmanuals.com/professional/geriatrics/approach-to-the-geriatric-patient/physical-changes-with-aging.]
Conflicting Evidence: “Heart rate, heart rate variability, and atrioventricular (AV) conduction were studied in 20 young (30 +/- 5 yr) and 19 older (69 +/- 7 yr) healthy men and women… Basal R-R intervals did not differ…” [N. Craft, J. B. Schwartz (1995). Effects of age on intrinsic heart rate, heart rate variability, and AV conduction in healthy humans. American Journal of Physiology-Heart and Circulatory Physiology. 268(4). https://doi.org/10.1152/ajpheart.1995.268.4.H1441.]
Heart Failure | Heart Rate +10 bpm
Moderately supported.
Direct Evidence: “Patients with HF frequently display a more rapid, shallow breathing pattern… Impaired CO is the hemodynamic feature that shows the strongest correlation with reduced respiratory muscles strength in HF.” Mechanism: Heart failure causes compensatory tachycardia to maintain cardiac output despite reduced stroke volume. [Hollenberg, S. M., & Parrillo, J. E. (2020). Altered Hemodynamics and End-Organ Damage in Heart Failure. Circulation, 142(7), 612-631. https://doi.org/10.1161/CIRCULATIONAHA.119.045409.]
Heart Failure | SpO2 -2%
Moderately supported.
Evidence: “Pulmonary congestion or edema [in HF causes] increased lung water content… [leading to] impaired gas transfer and V/Q mismatch.” [Frederik H. Verbrugge, MD, PhD, et al. (2020). Altered Hemodynamics and End-Organ Damage in Failure: Impact on the Lung and Kidney. Circulation, AHA Journals, 142(10). https://doi.org/10.1161/CIRCULATIONAHA.119.045409.]
This implies -2% is conservative estimate for pulmonary edema effects on SpO2.
Age >80 | SpO2 -1%
Weakly supported.
Indirect Evidence: Large COVID-19 cohort (N=8,770) showed consistent SpO2 decline with age: Age >75 years: SpO2 94% vs. Age 18-65: SpO2 97%. Limitations: No direct CKD-specific study; inferred from age effect.-1% is very conservative given 2-3% observed difference. [Rechtman, E., et al. (2020). Vital signs assessed in initial clinical encounters predict COVID-19 mortality in an NYC hospital system. Scientific Reports, 10, 21545. https://doi.org/10.1038/s41598-020-78392-1.]
Type 2 Diabetes | SBP +5 mmHg
Weakly supported.
Evidence: “Strong association between diabetes and hypertension in metabolic syndrome” but no specific +5 mmHg quantification. Limitation: Diabetic hypertension typically shows 10-20 mmHg increases in literature. Your +5 is either (a) very conservative, or (b) assumes diabetes WITHOUT concurrent hypertension diagnosis. [Silviu Stanciu, et al. (2023). Links between Metabolic Syndrome and Hypertension; The Relationship with the Current Antidiabetic Drugs. Metabolites, 13(1), 87. https://doi.org/10.3390/metabo13010087.]
Chronic Kidney Disease | SpO2 -1%
Weakly supported.
Mechanism (Inferred): CKS → fluid overload → pulmonary edema → ↓SpO2 (similar pathway to heart failure). Limitations: No direct CKD-to-SpO2 literature found. Plausible but not quantified. This implies -1% is reasonable but lacks explicit peer-reviewed support. [Rechtman, E., et al. (2020). Vital signs assessed in initial clinical encounters predict COVID-19 mortality in an NYC hospital system. Scientific Reports, 10, 21545. https://doi.org/10.1038/s41598-020-78392-1.]
**Heart Failure | SBP -10 mmHg 
Contradicted by evidence.
Direct Quote: “Among hospitalized patients with HFpEF, a discharge SBP level of less than 120 mmHg was associated with a significantly higher risk of 30-day, 1-year, and long-term all-cause mortality.” [Abuhasira, L., et al. (2018). Systolic Blood Pressure and Outcomes in Patients With Heart Failure With Preserved Ejection Fraction. JAMA, 3(4), 288–297. https://doi.org/10.1001/jamacardio.2017.5365.]
Direct Quote: “A discharge SBP <130 mmHg is associated with a higher risk of mortality and readmission… even SBP values between 110-129 mmHg are associated with poor outcomes in HFrEF.” [Kapoor, P. M., et al. (2024). Systolic Blood Pressure and Adverse Outcomes in Patients with Heart Failure and Reduced Ejection Fraction. NCBI, 73(24), 3054–3063. https://doi.org/10.1016/j.jacc.2019.04.022.]

Literature suggests Heart failure raises BP, especially as it often occurs with hypertension. Such interplay of factors make it more difficult to actually generate a script that replicates some patterns and trends. Hence, the goal of the data generation was just to create data that can be used to demonstrate and test time-series capabilities of LSTMs.

Key design choice: The train/test split is performed at the cohort level, before sequence generation. This ensures that the test set contains patient archetypes never seen during training, providing a rigorous evaluation of generalization.
2.3 Temporal Sequence Generation for Medical Conditions
For each patient profile, 60-second sequences (sampled at 1 Hz, yielding 60 timesteps) are generated for 13 distinct medical conditions:

Stable: Normal, healthy state
Monitor: Mildly abnormal vitals warranting observation
Heart_Attack: Acute myocardial infarction
Arrhythmia: Irregular cardiac rhythm
Heart_Failure: Acute decompensation
Hypoglycemia: Low blood sugar
Hyperglycemia_DKA: High blood sugar with ketoacidosis
Respiratory_Distress: Difficulty breathing, low oxygen
Sepsis: Systemic inflammatory response to infection
Stroke: Ischemic or hemorrhagic stroke
Shock: Circulatory collapse
Hypertensive_Crisis: Severely elevated blood pressure
Fall_Unconscious: Sudden loss of consciousness

Stable and Monitor States: Non-emergency states are generated using a bounded random walk algorithm. Several random "anchor points" are placed near the patient's baseline, and cubic interpolation creates a smooth curve passing through these points. This simulates natural physiological fluctuations while remaining within clinically acceptable ranges.

Emergency States: Emergency conditions are modeled using a multi-phase temporal progression system. Each condition has its own clinically-inspired progression that unfolds over the 60-second window. For example, a "classic STEMI" (ST-elevation myocardial infarction) is modeled in three phases:

Phase 1 (Ischemia, 15 seconds): Heart rate increases linearly from baseline, systolic pressure rises slightly
Phase 2 (Injury, 25 seconds): Sharp spike in heart rate, drop in SpO2 (representing oxygenation disturbance), systolic pressure plateaus then drops (representing cardiac muscle injury)
Phase 3 (Infarction, 20 seconds): Vital signs collapse—heart rate may become irregular, blood pressure plummets, SpO2 drops critically, representing irreversible damage

The pipeline includes multiple variations for each condition (e.g., slow_onset_nstemi, silent_mi_diabetic for heart attack) to increase data diversity and prevent the model from memorizing a single pattern per condition.
2.4 Data Validation
The resulting temporal sequences mimic actual behavior of vital signs during emergencies with low to moderate accuracy, when compared to a textual description of the actual behavior. The synthetic trends however, on being visualized, can be clearly and easily identified as to being synthetic data.

![Clinical Telemetry Gallery](./src/v2_clinical_gallery.png)
*Figure: Comprehensive visualization of 12 clinical emergency trajectories generated via SimGen v2.*

2.4.1 Myocardial Infarction (Heart Attack)
Figure 1. Sample Synthetic Data Simulating Heart Attack
![Heart Attack Pattern](./mermaid-diagram-2026-05-14-110611.png)

Acute myocardial infarction is modeled as an evolving failure of cardiac pump function with an early sympatho‑adrenergic surge followed by hemodynamic collapse. HR is ramped up into mild–moderate tachycardia in Phase 1 (pain/anxiety compensation), with an option for relative bradycardia in selected variants to mimic inferior-wall involvement of the conduction system. SBP initially rises slightly, then plateaus, and finally drops in Phase 3 as stroke volume falls. RR increases steadily due to pain, anxiety, and early pulmonary congestion, while SpO₂ drifts downward as simulated pulmonary edema impairs gas exchange. Temperature is kept near baseline over 60 seconds, because post‑infarction inflammatory fever typically appears hours to days later, not in the acute one‑minute horizon. [Mechanic, O. J., Grossman, S. A., & Azzouz, M. (2023). Acute Myocardial Infarction. StatPearls Publishing. https://www.ncbi.nlm.nih.gov/books/NBK459269/.]  [Thygesen, K., et al. (2019). Fourth universal definition of myocardial infarction. Circulation, European Heart Journal, 140(2), 128–151. https://academic.oup.com/eurheartj/article-abstract/40/3/226/5288651.] [Van Diepen, S., et al. (2017). Contemporary management of cardiogenic shock. Circulation, American Health Association Journals, 136(16), e232–e268. https://www.ahajournals.org/doi/full/10.1161/CIR.0000000000000525.]
2.4.2 Stroke (Ischemic or Hemorrhagic)
Figure 2. Sample Synthetic Data Simulating Stroke
![Stroke Pattern](./mermaid-diagram-2026-05-14-110829.png)

Stroke trajectories focus on cerebrovascular and autonomic responses rather than primary cardiopulmonary failure. BP is driven into a sustained hypertensive range throughout the sequence, reflecting both pre‑existing hypertension and reactive elevation as the body attempts to perfuse ischemic brain tissue. HR is allowed to be normal, mildly tachycardic, or irregular, capturing scenarios such as atrial fibrillation–related embolic strokes. RR is modeled as slightly increased and variably irregular, approximating disordered central drive or Cheyne–Stokes‑like patterns in severe brainstem involvement. SpO₂ is generally preserved, only modestly reduced if consciousness is simulated as impaired, and temperature is allowed a slow upward drift in some variants to approximate hypothalamic injury or post‑stroke hyperthermia. [​​Jauch, E. C., et al. (2013). Guidelines for the early management of patients with acute ischemic stroke. Stroke, American Health Association Journals, 44(3), 870–947. https://www.ahajournals.org/doi/full/10.1161/STR.0b013e318284056a.]


2.4.3 Sepsis
Figure 3. Sample Synthetic Data Simulating Sepsis
![Sepsis Pattern](./mermaid-diagram-2026-05-14-111131.png)

Sepsis is represented as a progressive distributive shock state with high metabolic demand. HR rises rapidly into marked tachycardia and remains elevated, reflecting compensation for reduced systemic vascular resistance. SBP and DBP show a gradual but persistent decline over the 60 seconds, modeling the evolution from normotension to hypotension as vasodilation and capillary leak worsen. RR is ramped steeply upward to reflect tachypnea and partial respiratory compensation for metabolic acidosis. SpO₂ is kept near baseline initially but trends downward in late segments in variants that approximate ARDS‑like involvement. Temperature is allowed to vary between high fever and relative hypothermia across different patient archetypes, capturing both classic hyperinflammatory presentations and blunted febrile responses in frail or severely septic patients. [Mahapatra, S., Heffner, A. C., & Huckleberry, Y. A. (2023). Septic Shock. StatPearls Publishing. https://www.ncbi.nlm.nih.gov/books/NBK430939/.] [Singer, M., et al. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis‑3). JAMA, 315(8), 801–810. https://jamanetwork.com/journals/jama/fullarticle/2492881.] [Kaukonen, K. M., et al. (2015). Systemic inflammatory response syndrome criteria in defining severe sepsis. New England Journal of Medicine, 372(17), 1629–1638. https://www.nejm.org/doi/full/10.1056/NEJMoa1415236.]
2.4.4 Diabetic Ketoacidosis (Hyperglycemia_DKA)
Figure 4. Sample Synthetic Data Simulating Hyperglycemia DKA
![DKA Pattern](./mermaid-diagram-2026-05-14-113153.png)

DKA trajectories emphasize metabolic derangement and dehydration rather than primary cardiopulmonary failure. Blood glucose is modeled as a monotonically rising or persistently very high curve across the entire window, distinguishing this class from others. HR is moderately tachycardic due to volume depletion and stress response, and BP is kept low‑normal or mildly hypotensive to reflect dehydration and reduced effective circulating volume. RR is configured as deep, regular tachypnea (Kussmaul‑like pattern) by increasing both rate and tidal amplitude surrogate, while SpO₂ is generally preserved because gas exchange remains structurally intact. Temperature is kept normal or slightly low, consistent with the fact that classic DKA is often normothermic despite possible underlying infection. [Kitabchi, A. E., et al. (2009). Hyperglycemic crises in adult patients with diabetes. Diabetes Care, 32(7), 1335–1343. https://pmc.ncbi.nlm.nih.gov/articles/PMC2699725/.] [Fayfman, M., Pasquel, F. J., & Umpierrez, G. E. (2017). Management of hyperglycemic crises. Medical Clinics of North America, 101(3), 587–606. https://www.medical.theclinics.com/article/S0025-7125(16)37404-1/abstract.]
2.4.5 Hypertensive Crisis
Figure 5. Sample Synthetic Data Simulating Hypertensive Crisis
![Hypertensive Crisis Pattern](./mermaid-diagram-2026-05-14-113319.png)

Hypertensive Crisis is modeled as a primarily hemodynamic emergency. SBP and DBP are driven sharply upward and held in a severely hypertensive range across the entire sequence (e.g., SBP > 180 mmHg), with relatively stable HR and SpO₂ to emphasize that the dominant abnormality is pressure overload rather than pump failure or hypoxia. RR is kept near baseline or mildly elevated to represent distress or early pulmonary congestion in some variants. Temperature remains normal. This pattern differentiates Hypertensive_Crisis from conditions like Sepsis or Shock, where hypotension and multi‑system compromise dominate. [Varon, J. (2008). Treatment of acute severe hypertension: current and newer agents. Drugs, Springer Nature Link, 68(3), 283–297. https://link.springer.com/article/10.2165/00003495-200868030-00003.] [van den Born, B. J., et al. (2019). ESC Council on Hypertension position document on hypertensive emergencies. European Heart Journal – Cardiovascular Pharmacotherapy, 5(1), 37–46. https://academic.oup.com/ehjcvp/article-abstract/5/1/37/5079054.] [Monica Aggarwal MD, Ijaz A. Khan MD (2006). Hypertensive Crisis: Hypertensive Emergencies and Urgencies. Cardiology Clinics, 24(1), 135-146. https://www.cardiology.theclinics.com/article/S0733-8651(05)00073-1/abstract.] 
2.4.6 Hypovolemic Shock (Shock)
Figure 6. Sample Synthetic Data Simulating Shock
![Shock Pattern](./mermaid-diagram-2026-05-14-113655.png)

Shock sequences represent a loss of circulating volume with progressive failure of compensatory mechanisms. HR ramps quickly into pronounced tachycardia and remains high throughout the 60 seconds as the primary compensatory mechanism. SBP and DBP initially remain near baseline or show a slight dip, then fall sharply in the latter part of the window to reflect decompensation. RR increases steadily to mimic tachypnea and respiratory compensation for metabolic acidosis. SpO₂ is initially preserved but is allowed to drop late as global perfusion fails. Temperature is modeled with a gradual downward drift in many variants to capture peripheral vasoconstriction and loss of thermal homeostasis in advanced shock. [Fair, K. A., & Balk, R. A. (2025). Hypovolemia and Hypovolemic Shock. StatPearls Publishing. https://www.ncbi.nlm.nih.gov/books/NBK513297/.] [Samuel M. Galvagno, Jr., DO, PhD, Jeffry T. Nahmias, MD, MHPEb, David A. Young, MD, MEd, MBA. American College of Surgeons. (2018). ATLS: Advanced Trauma Life Support (10th ed.).  Anesthesiology Clinics, 37(1), 13-32, https://www.anesthesiology.theclinics.com/article/S1932-2275(18)30096-X/abstract.]
2.4.7 Anaphylaxis (modeled within Shock / Respiratory_Distress variants)
Figure 7. Sample Synthetic Data Simulating Respiratory Distress
![Respiratory Distress Pattern](./mermaid-diagram-2026-05-14-113717.png)

Anaphylactic physiology is approximated in a subset of Shock and Respiratory_Distress variants by combining abrupt vasodilation and airway compromise. HR rises rapidly into severe tachycardia, while SBP and DBP drop precipitously early in the sequence. RR increases sharply, and SpO₂ shows a more rapid and deeper decline than in pure hypovolemic shock to reflect airway edema and bronchospasm. Temperature remains normal. These trajectories are parameterized as “fast‑onset shock with severe desaturation” and used to diversify the Shock and Respiratory_Distress classes without creating a separate label. [Campbell, R. L., et al. (2014). Emergency department diagnosis and treatment of anaphylaxis. Annals of Allergy, Asthma & Immunology, 113(6), 599–608. https://www.annallergy.org/article/S1081-1206(14)00743-1/fulltext.] [Cardona, V., et al. (2020). World Allergy Organization anaphylaxis guidance 2020. World Allergy Organization Journal, 13(10), 100472. https://www.sciencedirect.com/science/article/pii/S1939455120303756.]
2.4.8 Pneumonia and Respiratory Distress
Figure 8. Sample Synthetic Data Simulating Arrhythmia
![Arrhythmia Pattern](./mermaid-diagram-2026-05-14-120505.png)

Respiratory_Distress sequences capture a family of acute hypoxic respiratory failures, including pneumonia‑like patterns. RR is set to a high and often rising trajectory, with minimal recovery, reflecting sustained work of breathing. SpO₂ is configured to fall progressively from a near‑normal baseline to moderate or severe hypoxemia over 60 seconds, differentiating this class from cardiac emergencies where SpO₂ decline is often secondary. HR rises moderately due to hypoxia and stress, while BP is kept near normal unless a sepsis‑like overlap variant is generated. Temperature can be elevated in “infectious” respiratory variants (pneumonia‑like) and normal in non‑infectious ones (e.g., acute bronchospasm or pulmonary embolism‑like patterns). [Ignacio Martin-Loeches et al. (2023). ERS/ESICM/ESCMID/ALAT guidelines for the management of severe community-acquired pneumonia. Springer Nature Link, 49, 615-632. https://link.springer.com/article/10.1007/s00134-023-07033-8.] [National Library of Medicine, National Center for Biotechnology Information, USA (2001). BTS Guidelines for the management of community acquired pneumonia in adults. Thorax, 64(Suppl 3), iii1–iii55 (RR and SpO₂ criteria). https://pubmed.ncbi.nlm.nih.gov/11713364/.] [Fine, M. J., et al. (1997). A prediction rule to identify low-risk patients with community-acquired pneumonia. New England Journal of Medicine, 336(4), 243–250. https://cir.nii.ac.jp/crid/1573387449134099456.]
2.4.9 Pulmonary Embolism–like Patterns
Figure 9. Sample Synthetic Data Simulating Heart Failure
![Heart Failure Pattern](./mermaid-diagram-2026-05-14-121014.svg)

PE‑like dynamics are implemented as a subset of Respiratory_Distress and Shock trajectories. HR is rapidly tachycardic, RR is high with sudden onset, and SpO₂ drops abruptly rather than gradually, approximating the sudden V/Q mismatch typical of large pulmonary emboli. BP is either normal or falls abruptly in “massive PE” variants, while temperature remains normal or shows only a minimal rise. These patterns contribute to diversity within the Respiratory_Distress and Shock labels without introducing additional classes. [Konstantinides, S. V., et al. (2019). 2019 ESC Guidelines for the diagnosis and management of acute pulmonary embolism. European Heart Journal, 41(4), 543–603. https://academic.oup.com/eurheartj/article/41/4/543/5556136.] [Stein, P. D., et al. (2007). Clinical characteristics of patients with acute pulmonary embolism. CHEST, 131(2), 419–426. https://cir.nii.ac.jp/crid/1572261549899948160.]
2.4.10 Asthma Exacerbation–like Patterns
Figure 10. Sample Synthetic Data Simulating Fall Unconscious
![Fall Unconscious Pattern](./LSTM%20Sequence%20Processing-2026-05-14-051047.png)

Asthma‑like exacerbations are represented in a subset of Respiratory_Distress sequences with characteristic airflow limitation. RR is elevated with a pattern of prolonged expiration (implemented as asymmetric up‑down slopes in the RR surrogate), while SpO₂ declines modestly to reflect incomplete but impaired ventilation. HR is modestly tachycardic due to hypoxia, anxiety, and simulated beta‑agonist use. BP remains near normal, and temperature is fixed at baseline. These trajectories generate respiratory emergencies where hypoxia is present but hemodynamics remain relatively preserved. [Soriano, J. B., et al. (2017). Global, regional, and national deaths, prevalence, and disability-adjusted life years for chronic respiratory diseases. Lancet Respiratory Medicine, 5(9), 691–706. https://pmc.ncbi.nlm.nih.gov/articles/PMC5573769/.]
2.4.11 Stable and Monitor States
Figure 11. Sample Synthetic Data Simulating Stable

Figure 12. Sample Synthetic Data Simulating Monitor


Stable and Monitor trajectories provide non‑emergency baselines against which all emergency patterns must be distinguished. For Stable, all vitals are generated via bounded random walks around archetype‑specific baselines with low variance and no systematic trend, remaining within narrow normal ranges for the full 60 seconds. Monitor shares the same baseline structure but uses slightly larger random‑walk variance and occasional mild drifts (e.g., slowly rising HR, marginal BP elevation or depression, or gentle SpO₂ sag within normal or near‑normal limits). No coordinated multi‑system collapse is allowed in either state. This design ensures that the key difference between non‑emergency and emergency classes is the presence of coherent, multi‑variable, time‑locked patterns rather than isolated noisy deviations in a single vital. [Smith, G. B., et al. (2013). The ability of the National Early Warning Score (NEWS) to discriminate patients at risk of early cardiac arrest, unanticipated intensive care unit admission, and death. Resuscitation, 84(4), 465–470. https://www.sciencedirect.com/science/article/abs/pii/S0300957213000026.] [Romero‑Brufau, S., et al. (2025). Vital signs–only machine learning model for acute inpatient deterioration. Mayo Clinic Proceedings: Innovations, Quality & Outcomes, 9(5), 100663. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5219319.] [Kause, J., et al. (2004). A comparison of antecedents to cardiac arrests, deaths and emergency intensive care admissions in Australia and New Zealand, and the United Kingdom. Resuscitation, 62(3), 275–282. https://www.sciencedirect.com/science/article/abs/pii/S0300957204002473.]
2.4 Data Quality and Augmentation
Noise Injection: Realistic sensor noise is simulated by adding Gaussian noise with condition-specific standard deviations:

Heart Rate: σ = 2.0 bpm
Systolic BP: σ = 5.0 mmHg
Diastolic BP: σ = 3.0 mmHg
SpO2: σ = 0.5%
Temperature: σ = 0.1°C
Respiratory Rate: σ = 1.0 breaths/min
Blood Glucose: σ = 5.0 mg/dL

These values calibrate the model to handle realistic sensor variability.

Validation: All generated values are clipped to physiologically valid ranges (e.g., SpO2 ∈ [70, 100]%, HR ∈ [30, 220] bpm) to prevent impossible data points.

Augmentation: Non-stable sequences undergo one of three randomly selected augmentations:

Time Warping: Stretch/compress the time dimension (scale ∈ [0.9, 1.1]), simulating events that unfold faster or slower
Magnitude Scaling: Multiply entire sequence by random factor (scale ∈ [0.95, 1.05]), modeling sensor calibration variations
Baseline Shift: Add small offset (offset ∈ [−0.05, 0.05]), teaching the model to recognize patterns rather than absolute values

Refer to Appendix D for Gaussian Noise Standard Deviation and Augmentation Parameters.
2.5 Dataset Statistics
With default settings of 20 samples per cohort-class combination:

Total unique cohorts: 5 age bins × 2 genders × 3 activity levels × 8 conditions = 240 patient archetypes
Total training samples: 240 cohorts × 13 classes × 20 samples × 70% = ~43,680 sequences
Total test samples: 240 cohorts × 13 classes × 20 samples × 30% = ~18,720 sequences
Dataset balance: All classes equally represented (480 test samples per class)

A StandardScaler is fitted on training data only and applied to both train and test sets, ensuring test data remains a true unseen evaluation set.