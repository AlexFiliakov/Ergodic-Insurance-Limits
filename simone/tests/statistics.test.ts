import {
  calculateMean,
  calculateStandardDeviation,
  calculatePercentile,
  calculateEnsembleStatistics
} from '../src/utils/statistics';
import { SimulationResult } from '../src/models/types';

describe('Statistics', () => {
  describe('calculateMean', () => {
    test('should calculate mean correctly', () => {
      expect(calculateMean([1, 2, 3, 4, 5])).toBe(3);
      expect(calculateMean([10, 20, 30])).toBe(20);
    });

    test('should return 0 for empty array', () => {
      expect(calculateMean([])).toBe(0);
    });
  });

  describe('calculateStandardDeviation', () => {
    test('should calculate standard deviation correctly', () => {
      const values = [2, 4, 4, 4, 5, 5, 7, 9];
      const std = calculateStandardDeviation(values);
      expect(std).toBeCloseTo(2, 1);
    });

    test('should return 0 for single value', () => {
      expect(calculateStandardDeviation([5])).toBe(0);
    });
  });

  describe('calculatePercentile', () => {
    test('should calculate median (50th percentile) correctly', () => {
      const values = [1, 2, 3, 4, 5];
      expect(calculatePercentile(values, 50)).toBe(3);
    });

    test('should calculate 25th percentile correctly', () => {
      const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      expect(calculatePercentile(values, 25)).toBeCloseTo(3.25, 1);
    });

    test('should throw error for invalid percentile', () => {
      expect(() => calculatePercentile([1, 2, 3], -1)).toThrow();
      expect(() => calculatePercentile([1, 2, 3], 101)).toThrow();
    });
  });

  describe('calculateEnsembleStatistics', () => {
    test('should calculate ensemble statistics correctly', () => {
      const results: SimulationResult[] = [
        { step: 0, time: 0, value: 1 },
        { step: 1, time: 0.1, value: 2 },
        { step: 2, time: 0.2, value: 3 },
        { step: 3, time: 0.3, value: 4 },
        { step: 4, time: 0.4, value: 5 }
      ];

      const stats = calculateEnsembleStatistics(results);
      expect(stats.mean).toBe(3);
      expect(stats.min).toBe(1);
      expect(stats.max).toBe(5);
      expect(stats.std).toBeCloseTo(1.414, 2);
    });
  });
});
