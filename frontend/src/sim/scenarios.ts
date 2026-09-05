import type { Observation } from "../api/types";

export type ScenarioId = "stable" | "stress" | "sepsis" | "recovering";

export const SCENARIO_IDS: ScenarioId[] = [
  "stable",
  "stress",
  "sepsis",
  "recovering",
];

export const SCENARIO_LABELS: Record<ScenarioId, string> = {
  stable: "Stable",
  stress: "Stress",
  sepsis: "Sepsis",
  recovering: "Recovering",
};

const rand = (min: number, max: number): number => min + Math.random() * (max - min);

const randInt = (min: number, max: number): number => Math.floor(rand(min, max + 1));

const round2 = (value: number): number => Math.round(value * 100) / 100;

const round1 = (value: number): number => Math.round(value * 10) / 10;

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, value));

export interface PatientSimulatorOptions {
  patientId: string;
  age: number;
  scenario: ScenarioId;
  startIculos?: number;
}

export class PatientSimulator {
  private readonly patientId: string;
  private readonly age: number;
  private scenario: ScenarioId;
  private iculos: number;

  private hr: number;
  private o2sat: number;
  private sbp: number;
  private dbp: number;
  private map: number;
  private resp: number;
  private temp: number;

  private lactate: number;
  private wbc: number;
  private creatinine: number;
  private platelets: number;

  private stressHours: number;

  constructor({ patientId, age, scenario, startIculos = 1 }: PatientSimulatorOptions) {
    this.patientId = patientId;
    this.age = clamp(Math.round(age), 0, 120);
    this.scenario = scenario;
    this.iculos = startIculos;

    this.hr = rand(70, 85);
    this.o2sat = rand(97, 100);
    this.sbp = rand(110, 130);
    this.resp = rand(14, 18);
    this.temp = rand(36.5, 37.2);

    this.dbp = rand(65, 80);
    this.map = (this.sbp + 2 * this.dbp) / 3;

    this.lactate = rand(0.8, 1.5);
    this.wbc = rand(5.5, 10.5);
    this.creatinine = rand(0.6, 1.1);
    this.platelets = rand(180, 350);

    this.stressHours = scenario === "stress" ? randInt(2, 4) : 0;
  }

  private noise(value: number, amount: number): number {
    return value + rand(-amount, amount);
  }

  private applyScenario(): void {
    switch (this.scenario) {
      case "stable":
        this.hr = this.noise(this.hr, 2);
        this.o2sat = this.noise(this.o2sat, 0.3);
        this.sbp = this.noise(this.sbp, 3);
        this.resp = this.noise(this.resp, 1);
        this.temp = this.noise(this.temp, 0.15);
        break;

      case "stress":
        this.hr = this.noise(this.hr + 6, 2);
        this.resp = this.noise(this.resp + 2, 1);
        this.temp = this.noise(this.temp + 0.2, 0.1);
        this.sbp = this.noise(this.sbp + 4, 3);
        this.o2sat = this.noise(this.o2sat - 0.2, 0.2);
        this.stressHours -= 1;
        if (this.stressHours <= 0) {
          this.scenario = "recovering";
        }
        break;

      case "sepsis":
        this.hr = this.noise(this.hr + rand(1, 3), 1);
        this.resp = this.noise(this.resp + rand(0.5, 1.5), 0.5);
        this.temp = this.noise(this.temp + rand(0.05, 0.2), 0.05);
        this.sbp = this.noise(this.sbp - rand(0.5, 2), 1);
        this.o2sat = this.noise(this.o2sat - rand(0.1, 0.4), 0.2);
        this.lactate += rand(0.05, 0.2);
        this.wbc += rand(0.2, 0.8);
        break;

      case "recovering":
        this.hr = this.noise(Math.max(75, this.hr - 2), 1);
        this.resp = this.noise(Math.max(15, this.resp - 1), 0.5);
        this.temp = this.noise(Math.max(36.8, this.temp - 0.1), 0.05);
        this.sbp = this.noise(Math.min(120, this.sbp + 2), 1);
        this.o2sat = this.noise(Math.min(99, this.o2sat + 0.2), 0.2);
        this.lactate = Math.max(1.0, this.lactate - 0.1);
        this.wbc = Math.max(7.0, this.wbc - 0.3);
        break;
    }
  }

  private generateLabs(iculos: number): {
    Lactate: number | null;
    WBC: number | null;
    Creatinine: number | null;
    Platelets: number | null;
  } {
    const labs = {
      Lactate: null as number | null,
      WBC: null as number | null,
      Creatinine: null as number | null,
      Platelets: null as number | null,
    };

    if (iculos === 1) {
      labs.Lactate = round1(this.lactate);
      labs.WBC = round1(this.wbc);
      labs.Creatinine = round2(this.creatinine);
      labs.Platelets = Math.round(this.platelets);
    } else {
      if (iculos % 6 === 0) {
        labs.Lactate = round1(this.lactate);
      }
      if (iculos % 12 === 0) {
        labs.WBC = round1(this.wbc);
      }
      if (iculos % 24 === 0) {
        labs.Creatinine = round2(this.creatinine);
        labs.Platelets = Math.round(this.platelets);
      }
    }

    return labs;
  }

  generateNext(): Observation {
    this.applyScenario();

    this.dbp = clamp(this.sbp - rand(30, 45), 40, 100);
    this.map = (this.sbp + 2 * this.dbp) / 3;

    this.o2sat = clamp(this.o2sat, 70, 100);
    this.temp = clamp(this.temp, 35.0, 41.5);

    const labs = this.generateLabs(this.iculos);

    const obs: Observation = {
      PatientID: this.patientId,
      Age: this.age,
      ICULOS: this.iculos,
      HR: clamp(Math.round(this.hr), 20, 250),
      O2Sat: clamp(Math.round(this.o2sat), 0, 100),
      SBP: clamp(Math.round(this.sbp), 40, 300),
      MAP: clamp(Math.round(this.map), 20, 250),
      Resp: clamp(Math.round(this.resp), 0, 80),
      Temp: round1(this.temp),
      Lactate: labs.Lactate,
      WBC: labs.WBC,
      Creatinine: labs.Creatinine,
      Platelets: labs.Platelets,
    };

    this.iculos += 1;
    return obs;
  }
}