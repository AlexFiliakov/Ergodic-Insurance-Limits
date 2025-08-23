import { EnsembleStatistics, SimulationResult } from '../models/types';

export function calculateMean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, val) => sum + val, 0) / values.length;
}

export function calculateStandardDeviation(values: number[]): number {
  if (values.length <= 1) return 0;
  
  const mean = calculateMean(values);
  const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
  const variance = calculateMean(squaredDiffs);
  
  return Math.sqrt(variance);
}

export function calculateEnsembleStatistics(results: SimulationResult[]): EnsembleStatistics {
  const values = results.map(r => r.value);
  
  if (values.length === 0) {
    return {
      mean: 0,
      std: 0,
      min: 0,
      max: 0
    };
  }
  
  return {
    mean: calculateMean(values),
    std: calculateStandardDeviation(values),
    min: Math.min(...values),
    max: Math.max(...values)
  };
}

export function calculatePercentile(values: number[], percentile: number): number {
  if (values.length === 0) return 0;
  if (percentile < 0 || percentile > 100) {
    throw new Error('Percentile must be between 0 and 100');
  }
  
  const sorted = [...values].sort((a, b) => a - b);
  const index = (percentile / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index % 1;
  
  if (lower === upper) {
    return sorted[lower];
  }
  
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}