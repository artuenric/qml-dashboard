// src/App.tsx
import React, { useState } from 'react';
import './App.css';

const datasets = ['csv1', 'csv2', 'csv3','embraer'];
const algorithms = ['QSVM', 'QSVC', 'QNN', 'DT', 'NB', 'SVM'];
const environments = ['local', 'cloud'];
const clouds = ['IBM', 'AWS'];

function App() {
  const [dataset, setDataset] = useState('csv1');
  const [algorithm, setAlgorithm] = useState('DT');
  const [environment, setEnvironment] = useState<'local' | 'cloud'>('local');
  const [cloudProvider, setCloudProvider] = useState('IBM');
  const [token, setToken] = useState('');
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    try {
      const response = await fetch(`${apiUrl}/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          algorithm,
          dataset,
          cloud: cloudProvider,
          token
        })
      });

      const data = await response.json();
      if (data.status === 'ok') {
        setResult(data.resultado);
      } else {
        setResult(`Erro: ${data.mensagem}`);
      }
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(`Erro na requisição: ${err.message}`);
      } else {
        setError("Ocorreu um erro desconhecido");
      }
    } finally {
      setLoading(false);
    }
  };


  return (
    <div style={{ padding: '2rem', maxWidth: '600px', margin: '0 auto' }}>
      <h1>Quantum Machine Learning Executor</h1>

      <label>Dataset:</label>
      <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
        {datasets.map(d => <option key={d} value={d}>{d}</option>)}
      </select>

      <br /><br />

      <label>Algoritmo:</label>
      <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
        {algorithms.map(a => <option key={a} value={a}>{a}</option>)}
      </select>

      <br /><br />

      <label>Ambiente:</label>
      <select value={environment} onChange={(e) => setEnvironment(e.target.value as 'local' | 'cloud')}>
        {environments.map(env => <option key={env} value={env}>{env}</option>)}
      </select>

      {environment === 'cloud' && (
        <>
          <br />
          <label>Cloud Provider:</label>
          <select value={cloudProvider} onChange={(e) => setCloudProvider(e.target.value)}>
            {clouds.map(c => <option key={c} value={c}>{c}</option>)}
          </select>

          <br /><br />
          <label>Token:</label>
          <input type="text" value={token} onChange={(e) => setToken(e.target.value)} />
        </>
      )}

      <br /><br />
      <button onClick={handleSubmit}>Executar</button>

      {loading && <p>Executando...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {result && (
        <div style={{ marginTop: '2rem', padding: '1rem', border: '1px solid #ccc' }}>
          {result}
        </div>
      )}
    </div>
  );
}

export default App;

