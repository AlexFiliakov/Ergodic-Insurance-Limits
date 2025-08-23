import { Simulation } from '../src/core/simulation';
import { SimulationConfig } from '../src/models/types';

describe('Simulation', () => {
  let simulation: Simulation;
  let config: SimulationConfig;

  beforeEach(() => {
    config = {
      steps: 100,
      timeStep: 0.01,
      seed: 42
    };
    simulation = new Simulation(config);
  });

  test('should initialize with correct config', () => {
    const retrievedConfig = simulation.getConfig();
    expect(retrievedConfig.steps).toBe(100);
    expect(retrievedConfig.timeStep).toBe(0.01);
    expect(retrievedConfig.seed).toBe(42);
  });

  test('should run simulation for specified steps', () => {
    const results = simulation.run();
    expect(results.length).toBe(100);
  });

  test('should generate results with correct structure', () => {
    const results = simulation.run();
    const firstResult = results[0];

    expect(firstResult).toHaveProperty('step');
    expect(firstResult).toHaveProperty('time');
    expect(firstResult).toHaveProperty('value');
    expect(firstResult).toHaveProperty('metadata');
  });

  test('should calculate time correctly', () => {
    const results = simulation.run();

    for (let i = 0; i < results.length; i++) {
      expect(results[i].time).toBeCloseTo(i * config.timeStep, 10);
    }
  });
});
