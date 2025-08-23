export interface SimulationConfig {
  steps: number;
  timeStep: number;
  seed?: number;
  parameters?: Record<string, any>;
}

export interface SimulationResult {
  step: number;
  time: number;
  value: number;
  metadata?: Record<string, any>;
}

export interface TimeStep {
  current: number;
  delta: number;
  total: number;
}

export interface Distribution {
  mean: number;
  variance: number;
  skewness?: number;
  kurtosis?: number;
}

export interface EnsembleStatistics {
  mean: number;
  std: number;
  min: number;
  max: number;
  percentiles?: Record<number, number>;
}