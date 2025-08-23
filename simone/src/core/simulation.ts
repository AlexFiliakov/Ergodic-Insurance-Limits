import { SimulationConfig, SimulationResult, TimeStep } from '../models/types';

export class Simulation {
  private config: SimulationConfig;
  private currentStep: number = 0;
  private results: SimulationResult[] = [];

  constructor(config: SimulationConfig) {
    this.config = config;
  }

  public run(): SimulationResult[] {
    this.reset();

    while (this.currentStep < this.config.steps) {
      const result = this.step();
      this.results.push(result);
      this.currentStep++;
    }

    return this.results;
  }

  private step(): SimulationResult {
    const time = this.currentStep * this.config.timeStep;
    const value = this.computeValue(time);

    return {
      step: this.currentStep,
      time,
      value,
      metadata: {}
    };
  }

  private computeValue(time: number): number {
    return Math.sin(time) + Math.random() * 0.1;
  }

  private reset(): void {
    this.currentStep = 0;
    this.results = [];
  }

  public getResults(): SimulationResult[] {
    return [...this.results];
  }

  public getConfig(): SimulationConfig {
    return { ...this.config };
  }
}
