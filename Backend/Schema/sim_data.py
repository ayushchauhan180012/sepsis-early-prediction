import random

class Patient:
    def __init__(self, patient_id, age, state="stable"):
        self.patient_id = patient_id
        self.age = age
        self.state = state.lower()
        self.iculos = 1

        # Baseline vitals
        self.hr = random.uniform(70, 85)
        self.o2sat = random.uniform(97, 100)
        self.sbp = random.uniform(110, 130)
        self.resp = random.uniform(14, 18)
        self.temp = random.uniform(36.5, 37.2)

        self.dbp = random.uniform(65, 80)
        self.map = (self.sbp + 2 * self.dbp) / 3

        # Baseline labs
        self.lactate = random.uniform(0.8, 1.5)
        self.wbc = random.uniform(5.5, 10.5)
        self.creatinine = random.uniform(0.6, 1.1)
        self.platelets = random.uniform(180, 350)

        # Stress event duration
        self.stress_hours = random.randint(2, 4) if self.state == "stress" else 0

    def _noise(self, value, amount):
        return value + random.uniform(-amount, amount)

    def _stable(self):
        self.hr = self._noise(self.hr, 2)
        self.o2sat = self._noise(self.o2sat, 0.3)
        self.sbp = self._noise(self.sbp, 3)
        self.resp = self._noise(self.resp, 1)
        self.temp = self._noise(self.temp, 0.15)

    def _stress(self):
        self.hr = self._noise(self.hr + 6, 2)
        self.resp = self._noise(self.resp + 2, 1)
        self.temp = self._noise(self.temp + 0.2, 0.1)
        self.sbp = self._noise(self.sbp + 4, 3)
        self.o2sat = self._noise(self.o2sat - 0.2, 0.2)

        self.stress_hours -= 1
        if self.stress_hours <= 0:
            self.state = "recovering"

    def _sepsis(self):
        self.hr = self._noise(self.hr + random.uniform(1, 3), 1)
        self.resp = self._noise(self.resp + random.uniform(0.5, 1.5), 0.5)
        self.temp = self._noise(self.temp + random.uniform(0.05, 0.2), 0.05)

        self.sbp = self._noise(self.sbp - random.uniform(0.5, 2), 1)
        self.o2sat = self._noise(self.o2sat - random.uniform(0.1, 0.4), 0.2)

        self.lactate += random.uniform(0.05, 0.20)
        self.wbc += random.uniform(0.2, 0.8)

    def _recovering(self):
        self.hr = self._noise(max(75, self.hr - 2), 1)
        self.resp = self._noise(max(15, self.resp - 1), 0.5)
        self.temp = self._noise(max(36.8, self.temp - 0.1), 0.05)
        self.sbp = self._noise(min(120, self.sbp + 2), 1)
        self.o2sat = self._noise(min(99, self.o2sat + 0.2), 0.2)

        self.lactate = max(1.0, self.lactate - 0.1)
        self.wbc = max(7.0, self.wbc - 0.3)

    def _generate_labs(self):
        labs = {
            "Lactate": None,
            "WBC": None,
            "Creatinine": None,
            "Platelets": None
        }

        # Admission labs
        if self.iculos == 1:
            labs["Lactate"] = round(self.lactate, 2)
            labs["WBC"] = round(self.wbc, 2)
            labs["Creatinine"] = round(self.creatinine, 2)
            labs["Platelets"] = round(self.platelets, 2)

        else:
            if self.iculos % 6 == 0:
                labs["Lactate"] = round(self.lactate, 2)

            if self.iculos % 12 == 0:
                labs["WBC"] = round(self.wbc, 2)

            if self.iculos % 24 == 0:
                labs["Creatinine"] = round(self.creatinine, 2)
                labs["Platelets"] = round(self.platelets, 2)

        return labs

    def generate_tick(self):

        if self.state == "stable":
            self._stable()

        elif self.state == "stress":
            self._stress()

        elif self.state == "sepsis":
            self._sepsis()

        elif self.state == "recovering":
            self._recovering()

        self.dbp = max(40, min(100, self.sbp - random.uniform(30, 45)))
        self.map = (self.sbp + 2 * self.dbp) / 3

        self.o2sat = min(100, max(70, self.o2sat))
        self.temp = max(35.0, min(41.5, self.temp))

        labs = self._generate_labs()

        payload = {
            "PatientID": self.patient_id,
            "Age": round(self.age, 1),
            "ICULOS": self.iculos,
            "HR": round(self.hr, 2),
            "O2Sat": round(self.o2sat, 2),
            "SBP": round(self.sbp, 2),
            "MAP": round(self.map, 2),
            "Resp": round(self.resp, 2),
            "Temp": round(self.temp, 2),
            **labs
        }

        self.iculos += 1

        return payload

class SimulationEngine:
    def __init__(self):
        self.active_patients = {}
        self.next_patient_id = 1000

    def add_patient(self, age, state="stable"):
        patient_id = f"PT-{self.next_patient_id}"
        self.next_patient_id += 1

        self.active_patients[patient_id] = Patient(
            patient_id=patient_id,
            age=age,
            state=state
        )

        return patient_id

    def remove_patient(self, patient_id):
        self.active_patients.pop(patient_id, None)

    def generate_all_ticks(self):
        return [
            patient.generate_tick()
            for patient in self.active_patients.values()
        ]


if __name__ == "__main__":
    import json
    import time

    engine = SimulationEngine()
    STATE_STABLE = "stable"
    STATE_STRESS = "stress"
    STATE_SEPSIS = "sepsis"
    STATE_RECOVERING = "recovering"

    engine.add_patient(age=45, state=STATE_STABLE)
    engine.add_patient(age=70, state=STATE_STRESS)
    engine.add_patient(age=63, state=STATE_SEPSIS)

    print("Simulation Started...\n")

    try:
        while True:
            batch = engine.generate_all_ticks()

            print(json.dumps(batch, indent=2))
            print("-" * 60)

            # 1 minute = 1 simulated ICU hour
            time.sleep(60)

    except KeyboardInterrupt:
        print("\nSimulation stopped.")