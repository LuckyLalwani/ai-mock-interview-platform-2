import { useState } from 'react';
import { fuseEmotions, type FusionResult } from '@/lib/emotion';

export default function FusionTest() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState('');
  const [details, setDetails] = useState<FusionResult | null>(null);

  const testFusion = async () => {
    setLoading(true);
    setResult('');
    setDetails(null);

    try {
      const face = {
        anger: 0.02,
        disgust: 0.05,
        fear: 0.03,
        happiness: 0.65,
        neutral: 0.20,
        sadness: 0.02,
        surprise: 0.03,
      };

      const voice = {
        anger: 0.01,
        disgust: 0.02,
        fear: 0.05,
        happiness: 0.70,
        neutral: 0.18,
        sadness: 0.01,
        surprise: 0.03,
      };

      const text = {
        anger: 0.0,
        disgust: 0.0,
        fear: 0.01,
        happiness: 0.85,
        neutral: 0.12,
        sadness: 0.01,
        surprise: 0.01,
      };

      const fusion = await fuseEmotions(face, voice, text);

      if (!fusion.success) {
        throw new Error(fusion.error || 'Fusion failed.');
      }

      setDetails(fusion);
    } catch (error) {
      console.error(error);
      setResult(
        error instanceof Error ? error.message : 'Fusion failed.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="mx-auto max-w-2xl">
        <h1 className="mb-2 text-2xl font-bold text-gray-900">
          Emotion Fusion Test
        </h1>

        <p className="mb-6 text-gray-600">
          Test React (Local) &rarr; Google Cloud Run ML Service.
        </p>

        <button
          onClick={testFusion}
          disabled={loading}
          className="rounded-lg bg-primary-600 px-6 py-3 font-medium text-white disabled:opacity-50"
        >
          {loading ? 'Calling GCP API...' : 'Test Fusion on GCP'}
        </button>

        {result && (
          <div className="mt-6 rounded-xl border border-red-200 bg-red-50 p-5">
            <p className="font-medium text-red-700">{result}</p>
          </div>
        )}

        {details && (
          <div className="mt-6 space-y-4">
            <div className="rounded-xl border border-gray-200 bg-white p-6">
              <p className="mb-4 text-lg font-semibold text-gray-900">
                Live GCP Fusion Result
              </p>

              <div className="grid grid-cols-2 gap-4 text-sm text-gray-700">
                <div className="rounded-lg bg-gray-50 p-3">
                  <span className="text-gray-500">Dominant Emotion:</span>
                  <p className="text-base font-bold capitalize text-primary-600">
                    {details.dominant_emotion}
                  </p>
                </div>
                <div className="rounded-lg bg-gray-50 p-3">
                  <span className="text-gray-500">Overall Behavioral Score:</span>
                  <p className="text-base font-bold text-gray-900">
                    {details.overall_score?.toFixed(1)} / 100
                  </p>
                </div>
              </div>

              {details.behavioral_metrics && (
                <div className="mt-4 border-t pt-4">
                  <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-500">
                    Behavioral Metrics (From GCP Cloud Run)
                  </p>
                  <div className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-3">
                    <p><strong>Confidence:</strong> {details.behavioral_metrics.confidence?.toFixed(1)}%</p>
                    <p><strong>Composure:</strong> {details.behavioral_metrics.composure?.toFixed(1)}%</p>
                    <p><strong>Fluency:</strong> {details.behavioral_metrics.fluency?.toFixed(1)}%</p>
                    <p><strong>Engagement:</strong> {details.behavioral_metrics.engagement?.toFixed(1)}%</p>
                    <p><strong>Stress:</strong> {details.behavioral_metrics.stress?.toFixed(1)}%</p>
                    <p><strong>Adaptability:</strong> {details.behavioral_metrics.adaptability?.toFixed(1)}%</p>
                  </div>
                </div>
              )}
            </div>

            <div className="rounded-xl border border-gray-200 bg-gray-900 p-4 font-mono text-xs text-green-400">
              <p className="mb-2 text-gray-400">// Raw GCP JSON Response:</p>
              <pre className="overflow-x-auto whitespace-pre-wrap">
                {JSON.stringify(details, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}