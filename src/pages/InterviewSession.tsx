import { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { useNavigate, useParams } from '@/lib/router';
import { supabase } from '@/lib/supabase';
import { useAuth } from '@/lib/auth';
import { evaluateInterview } from '@/lib/ai';
import type { InterviewQuestion, Interview } from '@/types';
import {
  ArrowLeft,
  ArrowRight,
  Clock,
  AlertCircle,
  AlertTriangle,
  Loader2,
  Square,
  X,
  Mic,
  MicOff,
  Camera,
  VideoOff,
  CameraOff,
  RefreshCw,
  Activity,
  CheckCircle2,
  ShieldCheck,
  Volume2,
} from 'lucide-react';
import VoiceInterviewModal from '@/components/VoiceInterviewModal';
import AIInterviewerAvatar from '@/components/AIInterviewerAvatar';
import {
  predictFace,
  predictVoice,
  predictText,
  fuseEmotions,
  registerProctorFace,
  verifyProctorFace,
  transcribeAudio,
  type EmotionPrediction,
  type FusionResult,
  type ProctorResult,
} from '../lib/emotion';


interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message?: string;
}

interface SpeechRecognitionInstance {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start: () => void;
  stop: () => void;
  abort: () => void;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
}

declare global {
  interface Window {
    SpeechRecognition?: new () => SpeechRecognitionInstance;
    webkitSpeechRecognition?: new () => SpeechRecognitionInstance;
  }
}

/**
 * Merges newly transcribed text into existing answer without duplicating overlapping boundary words.
 */
function mergeTranscripts(existingText: string, newText: string): string {
  const cleanExisting = existingText.trim();
  const cleanNew = newText.trim();
  if (!cleanExisting) return cleanNew;
  if (!cleanNew) return cleanExisting;

  const normalize = (w: string) => w.toLowerCase().replace(/[.,!?;:'"()]/g, '');
  const existingWords = cleanExisting.split(/\s+/);
  const newWords = cleanNew.split(/\s+/);

  const maxCheck = Math.min(8, existingWords.length, newWords.length);
  let overlapCount = 0;

  for (let k = maxCheck; k >= 1; k--) {
    const existingSuffix = existingWords.slice(-k).map(normalize).join(' ');
    const newPrefix = newWords.slice(0, k).map(normalize).join(' ');
    if (existingSuffix === newPrefix) {
      overlapCount = k;
      break;
    }
  }

  if (overlapCount > 0) {
    const nonOverlapping = newWords.slice(overlapCount);
    if (nonOverlapping.length === 0) return cleanExisting;
    return cleanExisting + ' ' + nonOverlapping.join(' ');
  }

  return cleanExisting + ' ' + cleanNew;
}

function isSilenceHallucination(text: string): boolean {
  const t = text.trim().toLowerCase().replace(/[.,!?;:'"()]/g, '');
  if (!t) return true;
  if (t.includes('candidate speaking english') || t.includes('interview answer') || t.includes('technical speech')) {
    return true;
  }
  const hallucinations = new Set([
    'thank you',
    'thanks for watching',
    'thanks for listening',
    'bye',
    'you',
    'subtitles by',
    'blank audio',
    'so',
    'yes',
    'no',
    'im going to go to the next one',
    'i am going to go to the next one',
    'ill go to the next one',
    'going to go to the next one',
    'to the next one',
    '13 5',
    'i cant see',
    'and i cant wait to see the answer',
    'i cant wait to see the answer',
    'im sorry',
    'im not sure what im doing',
  ]);
  return hallucinations.has(t);
}

export default function InterviewSession() {
  const params = useParams('/interview/:id');
  const id = params?.id;
  const navigate = useNavigate();
  const { user } = useAuth();

  const [interview, setInterview] = useState<Interview | null>(null);
  const [questions, setQuestions] = useState<InterviewQuestion[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [showEndConfirm, setShowEndConfirm] = useState(false);
  const [ending, setEnding] = useState(false);
  const [voiceMode, setVoiceMode] = useState(false);
  const [voicePhase, setVoicePhase] = useState<
  'idle' | 'speaking' | 'listening'
>('idle');

const [transcript, setTranscript] = useState('');
const [interimTranscript, setInterimTranscript] = useState('');
const [faceEmotion, setFaceEmotion] =
  useState<EmotionPrediction | null>(null);

const [voiceEmotion, setVoiceEmotion] =
  useState<EmotionPrediction | null>(null);

const [textEmotion, setTextEmotion] =
  useState<EmotionPrediction | null>(null);

const [fusionResult, setFusionResult] =
  useState<FusionResult | null>(null);

const [proctorResult, setProctorResult] =
  useState<ProctorResult | null>(null);

const [proctorRegistered, setProctorRegistered] =
  useState(false);

const [proctorViolations, setProctorViolations] =
  useState(0);

const [isFullscreen, setIsFullscreen] = useState(false);
const [consecutiveNoFace, setConsecutiveNoFace] =
  useState(0);

const [consecutiveIdentityMismatch, setConsecutiveIdentityMismatch] =
  useState(0);

const [consecutiveMultipleFaces, setConsecutiveMultipleFaces] =
  useState(0);

const [isRecordingVoice, setIsRecordingVoice] =
  useState(false);

const mediaRecorderRef =
  useRef<MediaRecorder | null>(null);

const audioChunksRef =
  useRef<Blob[]>([]);

const recognitionRef =
  useRef<SpeechRecognitionInstance | null>(null);

const transcriptRef = useRef('');

  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [cameraPermission, setCameraPermission] = useState<
    'unknown' | 'granted' | 'denied' | 'loading'
  >('unknown');
  const [cameraStreamActive, setCameraStreamActive] = useState(false);
  const [cameraDevices, setCameraDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const startTimeRef = useRef<number>(0);
  const answerRef = useRef('');
  const handleEndRef = useRef<(() => Promise<void>) | null>(null);
  const terminationReasonRef = useRef<string | null>(null);
  // Guard flag: true while a face-emotion capture+inference cycle is in flight
  const isFaceCapturingRef = useRef(false);
  // Counter for consecutive retries when a frame is invalid (blurry, dark, etc.)
  const consecutiveInvalidRetriesRef = useRef(0);
  // Track fullscreen exits: 1st exit = warning alert, 2nd exit = end interview
  const [fullscreenExitCount, setFullscreenExitCount] = useState(0);
  const fullscreenExitCountRef = useRef(0);

  // Tab switch & Alt-Tab tracking: 1 warning allowed, 2nd switch terminates interview
  const [tabSwitchWarnings, setTabSwitchWarnings] = useState(0);
  const tabSwitchCountRef = useRef(0);
  const [showTabSwitchWarning, setShowTabSwitchWarning] = useState(false);

  // Dedicated termination overlay state with custom reason and Home navigation
  const [terminatedState, setTerminatedState] = useState<{
    isTerminated: boolean;
    reason: string;
  }>({ isTerminated: false, reason: '' });

  // Consecutive counters refs for proctoring checks
  const consecutiveNoFaceRef = useRef(0);
  const consecutiveIdentityMismatchRef = useRef(0);
  const consecutiveMultipleFacesRef = useRef(0);

  // Face registration states for instructions page
  const [registeringFace, setRegisteringFace] = useState(false);
  const [faceRegisterMessage, setFaceRegisterMessage] = useState<string | null>(null);

  // Track question speech so question is never re-read when clicking End or saving answers
  const lastSpokenIndexRef = useRef<number | null>(null);

  const [interviewStarted, setInterviewStarted] = useState(false);
  const interviewStartedRef = useRef(false);
  const [micEnabled, setMicEnabled] = useState(true);
  const micEnabledRef = useRef(true);
  const isEndingRef = useRef(false);
  const [micTestText, setMicTestText] = useState('');
  const [micTesting, setMicTesting] = useState(false);
  const isRecognizingRef = useRef(false);
  const isStartingRef = useRef(false);
  const shouldListenRef = useRef(false);
  const quickMicRecRef = useRef<any>(null);
  const finalChunksRef = useRef<string[]>([]);
  const lastFinalResultIndexRef = useRef<number>(-1);
  const sessionInterimRef = useRef('');
  const audioStreamRef = useRef<MediaStream | null>(null);

  // Single-instance Continuous Speech Pipeline (Groq Whisper)
  const isListeningActiveRef = useRef(false);
  const activeRecorderRef = useRef<MediaRecorder | null>(null);
  const segmentTimeoutRef = useRef<any>(null);
  const [isWhisperTranscribing, setIsWhisperTranscribing] = useState(false);

  const playWarningBeep = useCallback(() => {
    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(659.25, ctx.currentTime);
      osc.frequency.setValueAtTime(880, ctx.currentTime + 0.1);
      gain.gain.setValueAtTime(0.2, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.28);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.3);
    } catch {}
  }, []);

  // Active proctoring warning computation for eye-catching alert UI
  const activeProctorWarning = useMemo(() => {
    if (!interviewStarted || ending || isEndingRef.current || terminatedState.isTerminated) {
      return null;
    }

    if (consecutiveIdentityMismatch > 0) {
      return {
        level: 'critical' as const,
        title: 'UNAUTHORIZED / UNREGISTERED FACE DETECTED!',
        message: `Only the registered candidate is permitted. If an unauthorized face continues to be detected, your interview will be terminated (${consecutiveIdentityMismatch} of 3 warnings).`,
        count: consecutiveIdentityMismatch,
        max: 3,
      };
    }

    if (consecutiveMultipleFaces > 0) {
      return {
        level: 'critical' as const,
        title: 'MULTIPLE FACES DETECTED IN CAMERA!',
        message: `Please ensure you are alone in a private room. Multiple people detected in the camera frame (${consecutiveMultipleFaces} of 3 warnings).`,
        count: consecutiveMultipleFaces,
        max: 3,
      };
    }

    if (consecutiveNoFace > 0) {
      return {
        level: 'warning' as const,
        title: 'FACE NOT DETECTED IN CAMERA!',
        message: `Please position yourself directly in front of the camera with good lighting (${consecutiveNoFace} of 5 warnings).`,
        count: consecutiveNoFace,
        max: 5,
      };
    }

    return null;
  }, [
    interviewStarted,
    ending,
    terminatedState.isTerminated,
    consecutiveIdentityMismatch,
    consecutiveMultipleFaces,
    consecutiveNoFace,
  ]);

  useEffect(() => {
    interviewStartedRef.current = interviewStarted;
  }, [interviewStarted]);

  useEffect(() => {
    micEnabledRef.current = micEnabled;
  }, [micEnabled]);

  const enterFullscreen = useCallback(async () => {
    try {
      if (!document.fullscreenElement) {
        await document.documentElement.requestFullscreen();
      }
      setIsFullscreen(true);
    } catch (fullscreenError) {
      console.error('Fullscreen request failed:', fullscreenError);
      setIsFullscreen(false);
    }
  }, []);

  useEffect(() => {
    const handleFullscreenChange = () => {
      const fullscreen = Boolean(document.fullscreenElement);
      setIsFullscreen(fullscreen);

      if (!fullscreen) {
        // Only track exit when interview is in progress, not loading, started, and not already ending
        if (loading || ending || !id || !interviewStartedRef.current || isEndingRef.current) {
          return;
        }

        fullscreenExitCountRef.current += 1;
        const currentExits = fullscreenExitCountRef.current;
        setFullscreenExitCount(currentExits);

        if (currentExits === 1) {
          // Warning 1: Alert shown 1 time; do NOT terminate yet
          console.warn('Fullscreen Warning (1 of 2): Candidate exited fullscreen mode.');

          supabase
            .from('interview_proctor_events')
            .insert({
              interview_id: id,
              status: 'WARNING',
              similarity: null,
              faces_detected: null,
              is_match: null,
              message: 'Warning (1 of 2): Fullscreen mode was exited. One more exit will end interview.',
            })
            .then(({ error: warnErr }) => {
              if (warnErr) {
                console.error('Fullscreen warning save error:', warnErr);
              }
            });
        } else if (currentExits >= 2) {
          // Warning 2+: Exited fullscreen more than 1 time (2 times) -> Terminate interview immediately!
          console.error('Interview terminated: Exited fullscreen more than allowed limit (2 times).');
          terminationReasonRef.current = 'FULLSCREEN_EXIT';
          setTerminatedState({
            isTerminated: true,
            reason: 'Your interview has ended because fullscreen mode was exited more than the allowed limit (2 times).'
          });
          if (handleEndRef.current) {
            handleEndRef.current();
          }
        }
      }
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);

    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, [ending, loading, id]);

  // Alt-Tab & Screen switch tracking: 1 warning allowed, 2nd switch terminates interview
  useEffect(() => {
    const handleScreenOrTabSwitch = () => {
      if (loading || ending || !id || !interviewStartedRef.current || isEndingRef.current) {
        return;
      }

      tabSwitchCountRef.current += 1;
      const count = tabSwitchCountRef.current;
      setTabSwitchWarnings(count);

      if (count === 1) {
        console.warn('Screen switch warning (1 of 2): Candidate switched away (Alt+Tab or tab change).');
        setShowTabSwitchWarning(true);

        supabase
          .from('interview_proctor_events')
          .insert({
            interview_id: id,
            status: 'WARNING',
            similarity: null,
            faces_detected: null,
            is_match: null,
            message: 'Warning (1 of 2): Window or tab lost focus (Alt+Tab / Tab switch detected).',
          })
          .then(({ error: warnErr }) => {
            if (warnErr) console.error('Tab switch warning save error:', warnErr);
          });
      } else if (count >= 2) {
        console.error('Interview terminated: Screen switched more than allowed limit (2 times).');
        terminationReasonRef.current = 'TAB_SWITCH';
        setTerminatedState({
          isTerminated: true,
          reason: 'Your interview has ended because you switched away from the interview screen (Alt+Tab / tab change) more than the allowed limit (2 times).'
        });
        if (handleEndRef.current) {
          handleEndRef.current();
        }
      }
    };

    const handleVisibility = () => {
      if (document.hidden) {
        handleScreenOrTabSwitch();
      }
    };

    const handleBlur = () => {
      handleScreenOrTabSwitch();
    };

    document.addEventListener('visibilitychange', handleVisibility);
    window.addEventListener('blur', handleBlur);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibility);
      window.removeEventListener('blur', handleBlur);
    };
  }, [ending, loading, id]);

  useEffect(() => {
    answerRef.current = answer;
  }, [answer]);

  // Load interview and questions
  useEffect(() => {
    async function loadData() {
      if (!id || !user) return;

      setLoading(true);
      setError(null);

      try {
        const { data: interviewData, error: interviewError } =
          await supabase
            .from('interviews')
            .select('*')
            .eq('id', id)
            .maybeSingle();

        if (interviewError) {
          console.error('Interview loading error:', interviewError);
          setError('Failed to load interview.');
          setLoading(false);
          return;
        }

        if (!interviewData) {
          setError('Interview not found.');
          setLoading(false);
          return;
        }

        setInterview(interviewData as Interview);
        setCameraEnabled(Boolean(interviewData.camera_enabled));

        const { data: questionData, error: questionError } =
          await supabase
            .from('interview_questions')
            .select('*')
            .eq('interview_id', id)
            .order('question_index', { ascending: true });

        if (questionError) {
          console.error('Question loading error:', questionError);
          setError('Failed to load interview questions.');
          setLoading(false);
          return;
        }

        if (!questionData || questionData.length === 0) {
          setError('No questions were generated for this interview.');
          setLoading(false);
          return;
        }

        const loadedQuestions = questionData as InterviewQuestion[];

        setQuestions(loadedQuestions);
        const initialAns = loadedQuestions[0].answer_text ?? '';
        setAnswer(initialAns);
        answerRef.current = initialAns;
        finalChunksRef.current = initialAns.trim() ? [initialAns.trim()] : [];
        setTranscript(initialAns);

        const start = interviewData.started_at
          ? new Date(interviewData.started_at).getTime()
          : 0;

        startTimeRef.current = start;

        // If interview was already started previously (e.g. page refreshed during interview), resume directly
        if (interviewData.started_at && interviewData.status === 'in_progress') {
          setInterviewStarted(true);
        }
      } catch (loadError) {
        console.error('Interview load error:', loadError);
        setError('Failed to load interview. Please try again.');
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, [id, user]);

  // Timer
  useEffect(() => {
    if (!interviewStarted) return;

    const interval = setInterval(() => {
      if (startTimeRef.current) {
        setElapsed(
          Math.floor((Date.now() - startTimeRef.current) / 1000)
        );
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [interviewStarted]);

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;

    return `${m.toString().padStart(2, '0')}:${s
      .toString()
      .padStart(2, '0')}`;
  };

  const startVoiceEmotionRecording = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      return;
    }

    // Only one recorder at a time
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      return;
    }

    try {
      // Reuse the persistent audio stream; only create once
      if (!audioStreamRef.current || !audioStreamRef.current.active) {
        audioStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
      }

      const recorder = new MediaRecorder(audioStreamRef.current, { mimeType: 'audio/webm' });
      audioChunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        // Do NOT stop audio tracks — they stay alive for the whole interview
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        mediaRecorderRef.current = null;
        setIsRecordingVoice(false);

        if (audioBlob.size > 0) {
          try {
            const result = await predictVoice(audioBlob);
            if (result?.success) setVoiceEmotion(result);
          } catch {}
        }
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecordingVoice(true);
    } catch {}
  }, []);

useEffect(() => {
  if (
    !faceEmotion?.success ||
    !faceEmotion.predictions ||
    !voiceEmotion?.success ||
    !voiceEmotion.predictions
  ) {
    return;
  }

  const runFusion = async () => {
    try {
      const result = await fuseEmotions(
  faceEmotion?.predictions ?? null,
  voiceEmotion?.predictions ?? null,
  textEmotion?.predictions ?? null
);

      if (result.success) {
        setFusionResult(result);
      }
    } catch (fusionError) {
      console.error(
        'Emotion fusion error:',
        fusionError
      );
    }
  };

  runFusion();
}, [
  faceEmotion,
  voiceEmotion,
  textEmotion,
]);

  // Live text emotion inference debounced as candidate speaks or types
  useEffect(() => {
    if (!interviewStarted || !answer || answer.trim().length < 8 || ending || isEndingRef.current) {
      return;
    }

    const timer = setTimeout(async () => {
      try {
        const result = await predictText(answer.trim());
        if (result && result.success) {
          setTextEmotion(result);
        }
      } catch (err) {
        console.warn('Live text emotion error:', err);
      }
    }, 1200);

    return () => clearTimeout(timer);
  }, [answer, interviewStarted, ending]);
  const stopVoiceEmotionRecording = useCallback(() => {
  const recorder = mediaRecorderRef.current;

  if (!recorder) {
    return;
  }

  if (recorder.state !== 'inactive') {
    recorder.stop();
  }
}, []);
  const startListening = useCallback(async () => {
    if (isListeningActiveRef.current) {
      // Already running a speech capture loop! Prevent duplicate concurrent loops.
      return;
    }
    isListeningActiveRef.current = true;
    shouldListenRef.current = true;
    setVoicePhase('listening');

    if (!navigator.mediaDevices?.getUserMedia) {
      console.warn('Microphone access is not supported in this browser.');
      return;
    }

    try {
      if (!audioStreamRef.current || !audioStreamRef.current.active) {
        audioStreamRef.current = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
      }

      const stream = audioStreamRef.current;

      // Audio volume analyzer for silence detection
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      const audioCtx = new AudioCtx();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      const freqData = new Uint8Array(analyser.frequencyBinCount);

      // Stop any existing segment timer or active recorder
      if (segmentTimeoutRef.current) {
        clearTimeout(segmentTimeoutRef.current);
        segmentTimeoutRef.current = null;
      }
      if (activeRecorderRef.current && activeRecorderRef.current.state === 'recording') {
        try {
          activeRecorderRef.current.stop();
        } catch {}
        activeRecorderRef.current = null;
      }

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm';

      // Recursive continuous recorder: records 4.0s segments back-to-back
      const recordSegment = () => {
        if (!shouldListenRef.current || isEndingRef.current) return;

        try {
          const rec = new MediaRecorder(stream, { mimeType });
          activeRecorderRef.current = rec;
          const chunks: Blob[] = [];

          let maxVolumeDuringSegment = 0;
          const volumeCheckInterval = setInterval(() => {
            analyser.getByteFrequencyData(freqData);
            let sum = 0;
            for (let i = 0; i < freqData.length; i++) sum += freqData[i];
            const avg = sum / freqData.length;
            if (avg > maxVolumeDuringSegment) {
              maxVolumeDuringSegment = avg;
            }
          }, 100);

          rec.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) {
              chunks.push(e.data);
            }
          };

          rec.onstop = async () => {
            clearInterval(volumeCheckInterval);

            // Immediately start the next segment so audio capture has ZERO gaps
            if (shouldListenRef.current && !isEndingRef.current) {
              recordSegment();
            }

            // If user was silent during this 4-second segment, DO NOT call Whisper!
            // This prevents Whisper from hallucinating phantom text on room silence.
            if (maxVolumeDuringSegment < 8) {
              return;
            }

            if (chunks.length === 0) return;
            const blob = new Blob(chunks, { type: mimeType });
            if (blob.size < 3000) return;

            setIsWhisperTranscribing(true);
            try {
              const res = await transcribeAudio(blob);
              if (res.success && res.text) {
                const text = res.text.trim();
                if (text && !isSilenceHallucination(text)) {
                  // Direct non-closure update: prevents React StrictMode double execution!
                  const current = (answerRef.current || '').trim();
                  let toAppend = text;

                  // 1-word boundary deduplicator
                  if (current) {
                    const currentWords = current.split(/\s+/);
                    const newWords = toAppend.split(/\s+/);
                    const lastWord = currentWords[currentWords.length - 1]
                      .toLowerCase()
                      .replace(/[.,!?;:'"()]/g, '');
                    const firstWord = newWords[0]
                      .toLowerCase()
                      .replace(/[.,!?;:'"()]/g, '');

                    if (lastWord === firstWord && newWords.length > 1) {
                      toAppend = newWords.slice(1).join(' ');
                    } else if (lastWord === firstWord && newWords.length === 1) {
                      toAppend = '';
                    }
                  }

                  if (toAppend) {
                    const combined = current ? current + ' ' + toAppend : toAppend;
                    answerRef.current = combined;
                    setTranscript(combined);
                    setAnswer(combined);
                  }
                }
              }

              // Voice emotion analysis
              predictVoice(blob).then((vRes) => {
                if (vRes?.success) setVoiceEmotion(vRes);
              }).catch(() => {});
            } catch (err) {
              console.warn('Whisper segment transcription error:', err);
            } finally {
              setIsWhisperTranscribing(false);
            }
          };

          rec.start();

          // Stop after 4000ms
          segmentTimeoutRef.current = setTimeout(() => {
            if (rec.state === 'recording') {
              try {
                rec.stop();
              } catch {}
            }
          }, 4000);

        } catch (err) {
          console.warn('Error starting segment recorder:', err);
          if (shouldListenRef.current && !isEndingRef.current) {
            segmentTimeoutRef.current = setTimeout(recordSegment, 1000);
          }
        }
      };

      // Begin single segment loop
      recordSegment();
      setVoicePhase('listening');
    } catch (err) {
      console.error('Error starting continuous speech pipeline:', err);
      isListeningActiveRef.current = false;
      setVoicePhase('idle');
    }
  }, []);

  const stopListening = useCallback(() => {
    isListeningActiveRef.current = false;
    shouldListenRef.current = false;
    setVoicePhase('idle');
    setIsWhisperTranscribing(false);

    if (segmentTimeoutRef.current) {
      clearTimeout(segmentTimeoutRef.current);
      segmentTimeoutRef.current = null;
    }

    if (activeRecorderRef.current && activeRecorderRef.current.state === 'recording') {
      try {
        activeRecorderRef.current.stop();
      } catch {}
      activeRecorderRef.current = null;
    }
  }, []);

  const speakQuestion = useCallback(
  (questionText: string) => {
    if (!window.speechSynthesis) {
      console.warn('Speech synthesis is not supported.');
      return;
    }

    if (isEndingRef.current || ending || showEndConfirm || terminatedState.isTerminated) {
      window.speechSynthesis.cancel();
      return;
    }

    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(
      questionText
    );

    utterance.lang = 'en-US';
    utterance.rate = 0.95;
    utterance.pitch = 1;
    utterance.volume = 1;

    utterance.onstart = () => {
      setVoicePhase('speaking');
      // STOP MIC SO SPEAKERS PLAYING THE QUESTION ARE NEVER TRANSCRIBED!
      stopListening();
    };

    utterance.onend = () => {
      if (micEnabledRef.current && !isEndingRef.current && !showEndConfirm) {
        startListening();
      }
    };

    utterance.onerror = (event) => {
      console.error(
        'Speech synthesis error:',
        event
      );

      if (micEnabledRef.current && !isEndingRef.current && !showEndConfirm) {
        setVoicePhase('listening');
      } else {
        setVoicePhase('idle');
      }
    };

    window.speechSynthesis.speak(utterance);
  },
  [startListening, ending, showEndConfirm, terminatedState.isTerminated]
);

useEffect(() => {
  return () => {
    if (recognitionRef.current) {
      try {
        recognitionRef.current.abort();
      } catch {
        // Ignore cleanup errors.
      }
    }

    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach((track) => track.stop());
      audioStreamRef.current = null;
    }

    window.speechSynthesis?.cancel();
  };
}, []);

// Speak question effect: Strictly isolated to currentIndex change, NOT on answer save or questions array reference change
useEffect(() => {
  if (
    !interviewStarted ||
    !cameraEnabled ||
    loading ||
    questions.length === 0 ||
    ending ||
    isEndingRef.current ||
    showEndConfirm ||
    terminatedState.isTerminated
  ) {
    return;
  }

  // Only speak when currentIndex actually changes or upon starting Question 0
  if (lastSpokenIndexRef.current === currentIndex) {
    return;
  }

  const currentQuestion = questions[currentIndex];

  if (!currentQuestion?.question_text) {
    return;
  }

  lastSpokenIndexRef.current = currentIndex;

  const timer = window.setTimeout(() => {
    if (!isEndingRef.current && !ending && !showEndConfirm && !terminatedState.isTerminated) {
      speakQuestion(currentQuestion.question_text);
    }
  }, 500);

  return () => {
    window.clearTimeout(timer);
    window.speechSynthesis?.cancel();
  };
}, [
  interviewStarted,
  cameraEnabled,
  loading,
  currentIndex,
  ending,
  showEndConfirm,
  terminatedState.isTerminated,
  speakQuestion,
]);

  // Stop camera
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    setCameraStreamActive(false);
  }, []);

  // Start camera
  const startCamera = useCallback(async () => {
    if (!cameraEnabled) return;

    try {
      setCameraPermission('loading');

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          deviceId: selectedDeviceId
            ? { exact: selectedDeviceId }
            : undefined,
          facingMode: 'user',
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;

        try {
          await videoRef.current.play();
        } catch (playError) {
          console.warn('Video play error:', playError);
        }
      }

      setCameraStreamActive(true);
      setCameraPermission('granted');

      const devices = await navigator.mediaDevices.enumerateDevices();

      setCameraDevices(
        devices.filter((device) => device.kind === 'videoinput')
      );
    } catch (cameraError) {
      console.error('Camera access error:', cameraError);
      setCameraPermission('denied');
      setCameraStreamActive(false);
    }
  }, [cameraEnabled, selectedDeviceId]);

  const captureFaceEmotion = useCallback(async () => {
  if (!videoRef.current || !cameraStreamActive) {
    return;
  }

  // Drop-frame guard: if a previous capture+inference is still in
  // flight, skip this tick. This prevents an unbounded backlog when
  // the ML server or network is slower than the 1 FPS sampling interval.
  if (isFaceCapturingRef.current) {
    return;
  }

  isFaceCapturingRef.current = true;

  const video = videoRef.current;

  if (video.readyState < 2) {
    isFaceCapturingRef.current = false;
    return;
  }

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;

  const context = canvas.getContext('2d');

  if (!context) {
    isFaceCapturingRef.current = false;
    return;
  }

  context.drawImage(
    video,
    0,
    0,
    canvas.width,
    canvas.height
  );

  const blob = await new Promise<Blob | null>((resolve) => {
    canvas.toBlob(resolve, 'image/jpeg', 0.85);
  });

  if (!blob) {
    isFaceCapturingRef.current = false;
    return;
  }

  try {
    const result = await predictFace(blob);

    if (result.success) {
      // Reset retry counter on successful frame
      consecutiveInvalidRetriesRef.current = 0;
      // Valid prediction — update face emotion state which feeds fusion.
      setFaceEmotion(result);
    } else if (result.skipped) {
      // Frame was rejected by server-side quality validation (blur, lighting, etc.)
      // Do NOT set faceEmotion — invalid frames must never contribute to emotion aggregation
      console.warn(
        `[FaceEmotion] Frame rejected (${result.reason}) at ${new Date(
          (result.timestamp ?? Date.now() / 1000) * 1000
        ).toISOString()}`
      );
      // If frame is blurry or invalid, immediately capture another image instead of waiting 2 sec.
      if (consecutiveInvalidRetriesRef.current < 3) {
        consecutiveInvalidRetriesRef.current += 1;
        setTimeout(() => {
          if (cameraStreamActive && !isFaceCapturingRef.current) {
            captureFaceEmotion();
          }
        }, 80);
      } else {
        // Fall back to periodic interval after consecutive invalid retries
        consecutiveInvalidRetriesRef.current = 0;
      }
    }
  } catch (emotionError) {
    console.error(
      'Facial emotion prediction error:',
      emotionError
    );
  } finally {
    isFaceCapturingRef.current = false;
  }
}, [cameraStreamActive]);

const registerCandidateFace = useCallback(async (): Promise<boolean> => {
  if (!videoRef.current || !cameraStreamActive) {
    setFaceRegisterMessage('Please turn on your camera before registering your face.');
    return false;
  }

  const video = videoRef.current;

  if (video.readyState < 2) {
    setFaceRegisterMessage('Camera is initializing, please wait 1 second and try again.');
    return false;
  }

  setRegisteringFace(true);
  setFaceRegisterMessage('Scanning face and generating biometric embedding...');

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;

  const context = canvas.getContext('2d');

  if (!context) {
    setRegisteringFace(false);
    return false;
  }

  context.drawImage(
    video,
    0,
    0,
    canvas.width,
    canvas.height
  );

  const blob = await new Promise<Blob | null>((resolve) => {
    canvas.toBlob(resolve, 'image/jpeg', 0.9);
  });

  if (!blob) {
    setRegisteringFace(false);
    return false;
  }

  try {
    // Register candidate face
    const result = await registerProctorFace(blob);

    console.log('Proctor registration result:', result);

    if (result.success && result.status === 'VERIFIED') {
      setProctorRegistered(true);
      setFaceRegisterMessage('Face successfully registered and enrolled for proctoring verification! ✓');
      setProctorResult({
        status: 'VERIFIED',
        similarity: result.similarity,
        faces_detected: result.faces_detected,
        is_match: result.is_match,
        message: result.message,
      });
      return true;
    } else {
      setFaceRegisterMessage(result.message || 'Face not recognized. Please position your face clearly in the camera frame.');
      return false;
    }
  } catch (registrationError) {
    console.error(
      'Proctor registration error:',
      registrationError
    );
    setFaceRegisterMessage('Face registration service connection error. Please try again.');
    return false;
  } finally {
    setRegisteringFace(false);
  }
}, [
  cameraStreamActive,
]);

// Fallback registration if not registered yet when live interview begins
useEffect(() => {
  if (!interviewStarted || !cameraEnabled || !cameraStreamActive || proctorRegistered) {
    return;
  }

  const timer = window.setTimeout(() => {
    registerCandidateFace();
  }, 1500);

  return () => {
    window.clearTimeout(timer);
  };
}, [
  interviewStarted,
  cameraEnabled,
  cameraStreamActive,
  proctorRegistered,
  registerCandidateFace,
]);

useEffect(() => {
  if (
    !interviewStarted ||
    !cameraEnabled ||
    !cameraStreamActive ||
    !proctorRegistered ||
    ending ||
    isEndingRef.current ||
    terminatedState.isTerminated
  ) {
    return;
  }

  const interval = window.setInterval(async () => {
    if (!videoRef.current) {
      return;
    }

    const video = videoRef.current;

    if (video.readyState < 2) {
      return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const context = canvas.getContext('2d');

    if (!context) {
      return;
    }

    context.drawImage(
      video,
      0,
      0,
      canvas.width,
      canvas.height
    );

    const blob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob(resolve, 'image/jpeg', 0.85);
    });

    if (!blob) {
      return;
    }

    try {
      // Verify the current face against the registered candidate
      const result = await verifyProctorFace(blob);

      console.log('Proctor verification result:', result);

      setProctorResult(result);

      // Enforce limits:
      // - 3 warnings for wrong face (UNAUTHORIZED_FACE)
      // - 5 warnings for no face (NO_FACE)
      // - 3 warnings for multiple faces (MULTIPLE_FACES)
      if (result.status === 'VERIFIED') {
        consecutiveNoFaceRef.current = 0;
        consecutiveIdentityMismatchRef.current = 0;
        consecutiveMultipleFacesRef.current = 0;
        setConsecutiveNoFace(0);
        setConsecutiveIdentityMismatch(0);
        setConsecutiveMultipleFaces(0);
      } else if (result.status === 'NO_FACE') {
        consecutiveNoFaceRef.current += 1;
        const currentCount = consecutiveNoFaceRef.current;
        setConsecutiveNoFace(currentCount);
        playWarningBeep();

        if (id) {
          supabase
            .from('interview_proctor_events')
            .insert({
              interview_id: id,
              status: 'NO_FACE_WARNING',
              similarity: result.similarity ?? null,
              faces_detected: result.faces_detected ?? null,
              is_match: result.is_match ?? null,
              message: `No face warning (${currentCount} of 5)`,
            })
            .then(({ error: warnErr }) => {
              if (warnErr) console.error('Proctor event save error:', warnErr);
            });
        }

        if (currentCount >= 5) {
          console.error('Interview terminated: Face missing from frame 5 times.');
          terminationReasonRef.current = 'NO_FACE_LIMIT';
          setTerminatedState({
            isTerminated: true,
            reason: 'Your interview has ended because your face was not detected in the camera frame for 5 consecutive checks.'
          });
          if (handleEndRef.current) {
            handleEndRef.current();
          }
        }

        consecutiveIdentityMismatchRef.current = 0;
        consecutiveMultipleFacesRef.current = 0;
        setConsecutiveIdentityMismatch(0);
        setConsecutiveMultipleFaces(0);
      } else if (result.status === 'UNAUTHORIZED_FACE') {
        consecutiveIdentityMismatchRef.current += 1;
        const currentCount = consecutiveIdentityMismatchRef.current;
        setConsecutiveIdentityMismatch(currentCount);
        playWarningBeep();

        if (id) {
          supabase
            .from('interview_proctor_events')
            .insert({
              interview_id: id,
              status: 'WRONG_FACE_WARNING',
              similarity: result.similarity ?? null,
              faces_detected: result.faces_detected ?? null,
              is_match: result.is_match ?? null,
              message: `Wrong face warning (${currentCount} of 3)`,
            })
            .then(({ error: warnErr }) => {
              if (warnErr) console.error('Proctor event save error:', warnErr);
            });
        }

        if (currentCount >= 3) {
          console.error('Interview terminated: Unauthorized/mismatched face detected 3 times.');
          terminationReasonRef.current = 'WRONG_FACE_LIMIT';
          setTerminatedState({
            isTerminated: true,
            reason: 'Your interview has ended because an unregistered or mismatched face was detected for 3 consecutive checks.'
          });
          if (handleEndRef.current) {
            handleEndRef.current();
          }
        }

        consecutiveNoFaceRef.current = 0;
        consecutiveMultipleFacesRef.current = 0;
        setConsecutiveNoFace(0);
        setConsecutiveMultipleFaces(0);
      } else if (result.status === 'MULTIPLE_FACES') {
        consecutiveMultipleFacesRef.current += 1;
        const currentCount = consecutiveMultipleFacesRef.current;
        setConsecutiveMultipleFaces(currentCount);
        playWarningBeep();

        if (id) {
          supabase
            .from('interview_proctor_events')
            .insert({
              interview_id: id,
              status: 'MULTIPLE_FACES_WARNING',
              similarity: result.similarity ?? null,
              faces_detected: result.faces_detected ?? null,
              is_match: result.is_match ?? null,
              message: `Multiple faces warning (${currentCount} of 3)`,
            })
            .then(({ error: warnErr }) => {
              if (warnErr) console.error('Proctor event save error:', warnErr);
            });
        }

        if (currentCount >= 3) {
          console.error('Interview terminated: Multiple faces detected 3 times.');
          terminationReasonRef.current = 'MULTIPLE_FACES_LIMIT';
          setTerminatedState({
            isTerminated: true,
            reason: 'Your interview has ended because multiple faces were detected in the camera frame for 3 consecutive checks.'
          });
          if (handleEndRef.current) {
            handleEndRef.current();
          }
        }

        consecutiveNoFaceRef.current = 0;
        consecutiveIdentityMismatchRef.current = 0;
        setConsecutiveNoFace(0);
        setConsecutiveIdentityMismatch(0);
      }

      // Persist proctoring event
      if (id) {
        const { error: proctorSaveError } = await supabase
          .from('interview_proctor_events')
          .insert({
            interview_id: id,
            status: result.status,
            similarity: result.similarity ?? null,
            faces_detected: result.faces_detected ?? null,
            is_match: result.is_match ?? null,
            message: result.message ?? null,
          });

        if (proctorSaveError) {
          console.error(
            'Proctor event save error:',
            proctorSaveError
          );
        }
      }
    } catch (proctorError) {
      console.error(
        'Proctor verification error:',
        proctorError
      );
    }
  }, 3000);

  return () => {
    window.clearInterval(interval);
  };
}, [
  interviewStarted,
  cameraEnabled,
  cameraStreamActive,
  proctorRegistered,
  ending,
  terminatedState.isTerminated,
]);
  useEffect(() => {
  if (!interviewStarted || !cameraEnabled || !cameraStreamActive) {
    return;
  }

  // Sample one frame every 2 seconds.
  // If a frame is invalid (e.g. blurry), captureFaceEmotion immediately recaptures
  // a frame without waiting for the 2-second interval.
  const interval = window.setInterval(() => {
    captureFaceEmotion();
  }, 2000);

  return () => {
    window.clearInterval(interval);
  };
}, [
  interviewStarted,
  cameraEnabled,
  cameraStreamActive,
  captureFaceEmotion,
]);

  // Camera lifecycle
  useEffect(() => {
    if (!cameraEnabled) {
      stopCamera();
      setCameraPermission('unknown');
      return;
    }

    if (
      navigator.mediaDevices &&
      navigator.mediaDevices.getUserMedia
    ) {
      startCamera();
    }

    return () => stopCamera();
  }, [
    cameraEnabled,
    selectedDeviceId,
    startCamera,
    stopCamera,
  ]);

  // Ensure video element has the active stream
  useEffect(() => {
    if (
      videoRef.current &&
      streamRef.current &&
      cameraStreamActive
    ) {
      if (videoRef.current.srcObject !== streamRef.current) {
        videoRef.current.srcObject = streamRef.current;
        videoRef.current.play().catch((e) => console.warn('Video play error:', e));
      }
    }
  }, [cameraStreamActive, cameraEnabled, interviewStarted]);

  // Save answer
  const saveAnswer = useCallback(
  async (questionId: string, answerText: string) => {
    const trimmedAnswer = answerText.trim();

    let currentTextEmotion: EmotionPrediction | null = null;
    let currentFusionResult: FusionResult | null = fusionResult;

    if (trimmedAnswer) {
      try {
        const result = await predictText(trimmedAnswer);

        if (result.success) {
  currentTextEmotion = result;
  setTextEmotion(result);

  try {
    const fusion = await fuseEmotions(
      faceEmotion?.predictions ?? null,
      voiceEmotion?.predictions ?? null,
      result.predictions ?? null
    );

    if (fusion.success) {
      currentFusionResult = fusion;
      setFusionResult(fusion);
    }
  } catch (fusionError) {
    console.error('Answer fusion error:', fusionError);
  }
}
      } catch (textEmotionError) {
        console.error(
          'Text emotion prediction error:',
          textEmotionError
        );
      }
    }

    const { error: saveError } = await supabase
      .from('interview_questions')
      .update({
        answer_text: answerText,
        answer_status: trimmedAnswer ? 'answered' : 'unanswered',
        answered_at: trimmedAnswer
          ? new Date().toISOString()
          : null,

        // Emotion analysis captured for this answer
        face_emotion: faceEmotion,
        voice_emotion: voiceEmotion,
        text_emotion: currentTextEmotion,
        fusion_result: currentFusionResult,
      })
      .eq('id', questionId);

    if (saveError) {
      console.error('Answer save error:', saveError);
      throw new Error('Failed to save your answer.');
    }
  },
  [faceEmotion, voiceEmotion, fusionResult]
);

  // Dedicated camera handlers
  const handleStartCamera = useCallback(async () => {
    setCameraEnabled(true);
    await startCamera();
  }, [startCamera]);

  const handleStopCamera = useCallback(() => {
    setCameraEnabled(false);
    stopCamera();
  }, [stopCamera]);

  // Dedicated microphone handlers
  const handleStartMic = useCallback(() => {
    setMicEnabled(true);
    micEnabledRef.current = true;
    startListening();
  }, [startListening]);

  const handleStopMic = useCallback(() => {
    setMicEnabled(false);
    micEnabledRef.current = false;
    stopListening();
    setVoicePhase('idle');
  }, [stopListening]);

  // Quick mic test on instructions page using Groq Whisper
  const handleQuickMicTest = useCallback(async () => {
    if (micTesting) {
      if (quickMicRecRef.current) {
        try {
          quickMicRecRef.current.stop();
        } catch {}
        quickMicRecRef.current = null;
      }
      setMicTesting(false);
      return;
    }

    try {
      setMicTesting(true);
      setMicTestText('Listening... Speak a few words into your microphone now!');

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm';

      const rec = new MediaRecorder(stream, { mimeType });
      const testChunks: Blob[] = [];

      rec.ondataavailable = (ev) => {
        if (ev.data && ev.data.size > 0) {
          testChunks.push(ev.data);
        }
      };

      rec.onstop = async () => {
        setMicTesting(false);
        // Stop the isolated test tracks
        stream.getTracks().forEach((t) => t.stop());
        quickMicRecRef.current = null;

        const blob = new Blob(testChunks, { type: mimeType });
        if (blob.size > 1000) {
          setMicTestText('Transcribing with Groq Whisper...');
          try {
            const res = await transcribeAudio(blob);
            if (res.success && res.text && res.text.trim()) {
              setMicTestText(res.text.trim());
            } else {
              setMicTestText('No clear speech detected. Please speak louder and test again.');
            }
          } catch (err) {
            setMicTestText('Microphone captured audio, but transcription service was unreachable.');
          }
        } else {
          setMicTestText('No voice audio detected.');
        }
      };

      quickMicRecRef.current = rec;
      rec.start();

      // Automatically finish test after 3.8 seconds
      setTimeout(() => {
        if (rec.state === 'recording') {
          rec.stop();
        }
      }, 3800);
    } catch (micErr) {
      setMicTesting(false);
      setMicTestText('Microphone blocked. Please allow microphone permissions in your browser URL bar.');
    }
  }, [micTesting]);

  // Start interview from instructions page
  const handleStartInterview = useCallback(async () => {
    if (!proctorRegistered) {
      if (cameraStreamActive) {
        const registered = await registerCandidateFace();
        if (!registered) {
          setFaceRegisterMessage('Please position your face clearly in the camera and click "Register My Face" before starting.');
          return;
        }
      } else {
        setFaceRegisterMessage('Please turn on your camera and register your face before starting.');
        return;
      }
    }

    const now = Date.now();
    startTimeRef.current = now;
    setInterviewStarted(true);
    interviewStartedRef.current = true;

    if (id) {
      try {
        await supabase
          .from('interviews')
          .update({
            started_at: new Date(now).toISOString(),
            status: 'in_progress',
          })
          .eq('id', id);
      } catch (startErr) {
        console.warn('Could not update interview start time:', startErr);
      }
    }

    if (micEnabled) {
      startListening();
      if (cameraEnabled) {
        startVoiceEmotionRecording();
      }
    }
  }, [id, micEnabled, cameraEnabled, proctorRegistered, cameraStreamActive, registerCandidateFace, startListening, startVoiceEmotionRecording]);

  // Camera device switch
  const handleCameraDeviceSwitch = async () => {
    if (
      !navigator.mediaDevices ||
      !navigator.mediaDevices.enumerateDevices
    ) {
      return;
    }

    const devices =
      await navigator.mediaDevices.enumerateDevices();

    const videoDevices = devices.filter(
      (device) => device.kind === 'videoinput'
    );

    if (videoDevices.length <= 1) return;

    const currentIndex = videoDevices.findIndex(
      (device) => device.deviceId === selectedDeviceId
    );

    const nextDevice =
      videoDevices[
        (currentIndex + 1) % videoDevices.length
      ];

    setSelectedDeviceId(nextDevice.deviceId);
  };

  // Navigate between questions
  const goToQuestion = async (index: number) => {
    if (index < 0 || index >= questions.length || saving) {
      return;
    }
    stopListening();
    stopVoiceEmotionRecording();
    window.speechSynthesis?.cancel();

    // Reset speech transcripts for the newly selected question
    transcriptRef.current = '';
    setTranscript('');
    setInterimTranscript('');

    setSaving(true);
    setError(null);

    try {
      const currentQ = questions[currentIndex];

      if (currentQ) {
        await saveAnswer(currentQ.id, answerRef.current);
      }

      const updated = [...questions];

      if (updated[currentIndex]) {
        updated[currentIndex] = {
          ...updated[currentIndex],
          answer_text: answerRef.current,
          answer_status: answerRef.current.trim()
            ? 'answered'
            : 'unanswered',
          answered_at: answerRef.current.trim()
            ? new Date().toISOString()
            : null,
        };
      }

      setQuestions(updated);
      setCurrentIndex(index);
      const nextAnswer = updated[index].answer_text ?? '';
      setAnswer(nextAnswer);
      answerRef.current = nextAnswer;
      finalChunksRef.current = nextAnswer.trim() ? [nextAnswer.trim()] : [];
      lastFinalResultIndexRef.current = -1;
      transcriptRef.current = nextAnswer;
      sessionInterimRef.current = '';
      setTranscript(nextAnswer);
      setInterimTranscript('');

      // Reset speech pipeline for new question
      setIsWhisperTranscribing(false);
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : 'Failed to save your answer.'
      );
    } finally {
      setSaving(false);
      if (micEnabledRef.current && !isEndingRef.current && !showEndConfirm) {
        startListening();
      }
    }
  };

  // Finish interview and request AI evaluation
  const handleEnd = async () => {
    if (!id || !interview || ending) return;

    isEndingRef.current = true;
    setEnding(true);
    setError(null);
    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach((track) => track.stop());
      audioStreamRef.current = null;
    }

    window.speechSynthesis?.cancel();
    stopListening();
    stopCamera();
    stopVoiceEmotionRecording();

    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach((track) => track.stop());
      audioStreamRef.current = null;
    }

    try {
      // Save current answer first
      const currentQ = questions[currentIndex];

      if (currentQ) {
        await saveAnswer(currentQ.id, answerRef.current);

        const updatedQuestions = [...questions];

        updatedQuestions[currentIndex] = {
          ...updatedQuestions[currentIndex],
          answer_text: answerRef.current,
          answer_status: answerRef.current.trim()
            ? 'answered'
            : 'unanswered',
          answered_at: answerRef.current.trim()
            ? new Date().toISOString()
            : null,
        };

        setQuestions(updatedQuestions);
      }
      stopListening();
      window.speechSynthesis.cancel();
      // Stop camera before leaving the interview
      stopCamera();
      if (terminationReasonRef.current) {
  const reason = terminationReasonRef.current;

  await supabase
    .from('interview_proctor_events')
    .insert({
      interview_id: id,
      status: reason,
      similarity: null,
      faces_detected: null,
      is_match: null,
      message:
        reason === 'FULLSCREEN_EXIT'
          ? 'Interview terminated: Fullscreen mode was exited more than allowed limit (2 times).'
          : 'Interview terminated because the interview tab lost visibility.',
    });

  terminationReasonRef.current = null;
}
      // Calculate interview duration
      const durationSeconds = Math.floor(
        (Date.now() - startTimeRef.current) / 1000
      );

      // Mark interview as completed
      const { error: completeError } = await supabase
        .from('interviews')
        .update({
          status: 'completed',
          ended_at: new Date().toISOString(),
          duration_seconds: durationSeconds,
        })
        .eq('id', id);

      if (completeError) {
        console.error(
          'Interview completion error:',
          completeError
        );
      }

      // Ask the secure Edge Function to evaluate the interview.
      // OpenRouter is called only from the server.
      try {
        await evaluateInterview(id);
      } catch (evalError) {
        console.warn('AI evaluation warning (continuing to summary):', evalError);
      }

      navigate(`/summary/${id}`, { replace: true });
    } catch (endError) {
      console.error('Interview completion error:', endError);
      // Ensure candidate is navigated to summary even if a save step threw an error
      navigate(`/summary/${id}`, { replace: true });
    }
  };
  handleEndRef.current = handleEnd;

  if (loading) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <div className="text-center">
          <Loader2 className="mx-auto h-8 w-8 animate-spin text-primary-600" />
          <p className="mt-3 text-sm text-gray-500">
            Loading interview...
          </p>
        </div>
      </div>
    );
  }

  if (error && !interview) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center px-4">
        <div className="text-center">
          <AlertCircle className="mx-auto h-10 w-10 text-error-400" />

          <p className="mt-3 text-base font-medium text-gray-900">
            {error}
          </p>

          <button
            onClick={() => navigate('/dashboard')}
            className="btn-secondary mt-6"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  if (questions.length === 0) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center px-4">
        <div className="text-center">
          <AlertCircle className="mx-auto h-10 w-10 text-error-400" />

          <p className="mt-3 text-base font-medium text-gray-900">
            No questions were generated for this interview.
          </p>

          <button
            onClick={() => navigate('/dashboard')}
            className="btn-secondary mt-6"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const currentQ = questions[currentIndex];
  const progress =
    ((currentIndex + 1) / questions.length) * 100;

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-gray-50 animate-fade-in">
    {/* Fullscreen requirement modal */}
    {!isFullscreen && (
      <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/95 backdrop-blur-sm">
        <div className="max-w-md px-6 text-center text-white">
          {fullscreenExitCount >= 2 ? (
            <div>
              <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-red-600/20 text-red-500">
                <AlertCircle className="h-8 w-8" />
              </div>
              <h2 className="mb-3 text-2xl font-semibold text-red-400">
                Interview Terminated
              </h2>
              <p className="mb-6 text-gray-300">
                Your interview has ended because fullscreen mode was exited more than 2 times.
              </p>
              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <button
                  type="button"
                  onClick={() => navigate('/dashboard', { replace: true })}
                  className="rounded-lg bg-gray-700 hover:bg-gray-600 px-5 py-2.5 font-medium text-white transition"
                >
                  Go to Home Screen
                </button>
                <button
                  type="button"
                  onClick={() => navigate(`/summary/${id}`, { replace: true })}
                  className="rounded-lg bg-blue-600 hover:bg-blue-700 px-5 py-2.5 font-medium text-white transition"
                >
                  View Summary
                </button>
              </div>
            </div>
          ) : fullscreenExitCount === 1 ? (
            <div>
              <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-amber-600/20 text-amber-400">
                <AlertCircle className="h-8 w-8" />
              </div>
              <h2 className="mb-3 text-2xl font-semibold text-amber-300">
                Warning: Fullscreen Exited (1 of 2)
              </h2>
              <p className="mb-6 text-gray-300">
                You have exited fullscreen mode. This interview requires fullscreen mode. Exiting fullscreen a second time will end the interview.
              </p>
              <button
                onClick={enterFullscreen}
                className="rounded-lg bg-amber-600 px-6 py-3 font-medium text-white transition hover:bg-amber-700"
              >
                Return to Fullscreen
              </button>
            </div>
          ) : (
            <div>
              <h2 className="mb-3 text-2xl font-semibold">
                Enter Fullscreen Mode
              </h2>
              <p className="mb-6 text-gray-400">
                This interview must be completed in fullscreen mode.
                Please enter fullscreen to continue.
              </p>
              <button
                onClick={enterFullscreen}
                className="rounded-lg bg-blue-600 px-6 py-3 font-medium text-white transition hover:bg-blue-700"
              >
                Enter Fullscreen
              </button>
            </div>
          )}
        </div>
      </div>
    )}

    {/* Dedicated Screen Switch / Alt-Tab Warning Modal (Warning 1 of 2) */}
    {showTabSwitchWarning && (
      <div className="fixed inset-0 z-[99998] flex items-center justify-center bg-black/80 p-4 backdrop-blur-sm animate-fade-in">
        <div className="card max-w-md p-6 text-center bg-white shadow-xl rounded-2xl border border-amber-300 animate-slide-up">
          <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-amber-50 text-amber-600">
            <AlertCircle className="h-7 w-7" />
          </div>
          <h3 className="text-lg font-bold text-amber-800 mb-2">
            Warning: Screen Switched (1 of 2)
          </h3>
          <p className="text-xs text-gray-600 mb-5 leading-relaxed">
            You switched away from the interview screen (Alt+Tab or tab change detected). Please keep this interview window active and focused. Switching away a second time will immediately terminate your interview.
          </p>
          <button
            type="button"
            onClick={() => setShowTabSwitchWarning(false)}
            className="w-full rounded-lg bg-amber-600 hover:bg-amber-700 px-4 py-2.5 text-sm font-semibold text-white transition"
          >
            I Understand, Continue Interview
          </button>
        </div>
      </div>
    )}

    {/* Dedicated Termination Overlay (Screen switch, fullscreen exit, or proctor violations) */}
    {terminatedState.isTerminated && (
      <div className="fixed inset-0 z-[99999] flex items-center justify-center bg-black/90 p-4 backdrop-blur-md animate-fade-in">
        <div className="card max-w-lg p-6 sm:p-8 text-center bg-white shadow-2xl rounded-2xl animate-slide-up border border-error-200">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-error-50 text-error-600">
            <AlertCircle className="h-10 w-10" />
          </div>

          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            Interview Terminated
          </h2>

          <p className="text-sm text-gray-600 mb-6 leading-relaxed">
            {terminatedState.reason}
          </p>

          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <button
              type="button"
              onClick={() => navigate('/dashboard', { replace: true })}
              className="btn-secondary flex-1 py-3 justify-center text-sm font-semibold"
            >
              Go to Home Screen
            </button>

            <button
              type="button"
              onClick={() => navigate(`/summary/${id}`, { replace: true })}
              className="btn-primary flex-1 py-3 justify-center text-sm font-semibold bg-primary-600 hover:bg-primary-700"
            >
              View Interview Summary
            </button>
          </div>
        </div>
      </div>
    )}

      {voiceMode && (
        <VoiceInterviewModal
          questions={questions.map((q) => ({
            id: q.id,
            question_text: q.question_text,
            question_index: q.question_index,
            category: q.category,
          }))}
          onClose={() => setVoiceMode(false)}
          onComplete={async (voiceAnswers) => {
            setVoiceMode(false);
            setEnding(true);
            setError(null);

            try {
              // Save voice answers
              const updatePromises = questions.map(async (q) => {
                const ans = voiceAnswers[q.id] ?? '';
                const trimmedAnswer = ans.trim();

                let textEmotion = null;
                let currentFusionResult = null;

                if (trimmedAnswer) {
                  try {
                    const textResult = await predictText(trimmedAnswer);

                    if (textResult.success) {
                      textEmotion = textResult;

                      try {
                        const fusion = await fuseEmotions(
                          faceEmotion?.predictions ?? null,
                          voiceEmotion?.predictions ?? null,
                          textResult.predictions ?? null
                        );

                        if (fusion.success) {
                          currentFusionResult = fusion;
                        }
                      } catch (fusionError) {
                        console.error(
                          'Voice answer fusion error:',
                          fusionError
                        );
                      }
                    }
                  } catch (textEmotionError) {
                    console.error(
                      'Voice answer text emotion error:',
                      textEmotionError
                    );
                  }
                }

                return supabase
                  .from('interview_questions')
                  .update({
                    answer_text: ans,
                    answer_status: trimmedAnswer
                      ? 'answered'
                      : 'unanswered',
                    answered_at: trimmedAnswer
                      ? new Date().toISOString()
                      : null,
                    face_emotion: faceEmotion,
                    voice_emotion: voiceEmotion,
                    text_emotion: textEmotion,
                    fusion_result: currentFusionResult,
                  })
                  .eq('id', q.id);
              });

              const results = await Promise.all(updatePromises);
              const failedUpdate = results.find((result) => result.error);

              if (failedUpdate?.error) {
                throw new Error('Failed to save voice answers.');
              }

              // Stop camera if active
              stopCamera();

              // Calculate duration
              const durationSeconds = Math.floor(
                (Date.now() - startTimeRef.current) / 1000
              );

              // Complete interview
              const { error: completeError } =
                await supabase
                  .from('interviews')
                  .update({
                    status: 'completed',
                    ended_at: new Date().toISOString(),
                    duration_seconds: durationSeconds,
                  })
                  .eq('id', id!);

              if (completeError) {
                console.error('Completion error:', completeError);
              }

              // AI evaluation
              try {
                await evaluateInterview(id!);
              } catch (evalError) {
                console.warn('Voice AI evaluation warning (continuing to summary):', evalError);
              }

              navigate(`/summary/${id}`, {
                replace: true,
              });
            } catch (voiceError) {
              console.error(
                'Voice interview completion error:',
                voiceError
              );

              navigate(`/summary/${id}`, {
                replace: true,
              });
            }
          }}
        />
      )}

      {/* Top bar */}
      {interviewStarted && (
        <div className="sticky top-0 z-30 border-b border-gray-200 bg-white/90 backdrop-blur-md">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between py-3">
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium text-gray-900">
                  Question {currentIndex + 1} of {questions.length}
                </span>

                <span className="text-xs text-gray-400">·</span>

                <span className="text-sm text-gray-500 capitalize">
                  {currentQ.category?.replace('_', ' ') ||
                    currentQ.topic ||
                    'general'}
                </span>
              </div>

              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5 rounded-lg bg-gray-100 px-3 py-1.5">
                  <Clock className="h-4 w-4 text-gray-500" />
                  <span className="text-sm font-mono font-medium text-gray-700">
                    {formatTime(elapsed)}
                  </span>
                </div>

                {cameraEnabled && (
                  <span className="inline-flex items-center gap-1.5 rounded-lg bg-primary-50 px-3 py-1.5 text-sm font-medium text-primary-700">
                    <Mic className="h-3.5 w-3.5" />
                    Voice Active
                  </span>
                )}

                <button
                  onClick={() => {
                    window.speechSynthesis?.cancel();
                    stopListening();
                    setShowEndConfirm(true);
                  }}
                  disabled={ending}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-error-200 bg-white px-3 py-1.5 text-sm font-medium text-error-600 transition-colors hover:bg-error-50 disabled:opacity-50"
                >
                  <Square className="h-3.5 w-3.5" />
                  End
                </button>
              </div>
            </div>

            {/* Progress bar */}
            <div className="h-1 w-full rounded-full bg-gray-100">
              <div
                className="h-1 rounded-full bg-primary-600 transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      {!interviewStarted ? (
        /* Pre-Interview Instructions, Device & Face Registration Check Screen */
        <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8 animate-fade-in">
          <div className="mb-8 text-center">
            <div className="inline-flex items-center gap-2 rounded-full bg-primary-50 px-3 py-1 text-xs font-semibold text-primary-700 mb-3 border border-primary-200">
              <ShieldCheck className="h-4 w-4" />
              Secure AI Interview Environment
            </div>
            <h1 className="text-2xl font-bold text-gray-900 sm:text-3xl">
              Pre-Interview Setup & Verification
            </h1>
            <p className="mt-2 text-sm text-gray-600 max-w-xl mx-auto">
              Please register your face for identity verification, test your devices, and review the guidelines before starting.
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            {/* Left: Device & Face Registration Checks */}
            <div className="space-y-4">
              {/* Camera & Face Registration Card */}
              <div className="card p-5">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Camera className="h-4 w-4 text-primary-600" />
                    <span className="text-sm font-semibold text-gray-900">Camera & Face Registration</span>
                  </div>
                  {proctorRegistered ? (
                    <span className="inline-flex items-center gap-1 text-xs font-medium text-green-700 bg-green-50 px-2.5 py-0.5 rounded-full border border-green-200">
                      <CheckCircle2 className="h-3 w-3" /> Face Enrolled
                    </span>
                  ) : cameraStreamActive ? (
                    <span className="inline-flex items-center gap-1 text-xs font-medium text-amber-700 bg-amber-50 px-2.5 py-0.5 rounded-full border border-amber-200">
                      <AlertCircle className="h-3 w-3" /> Not Enrolled
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-xs font-medium text-gray-500 bg-gray-100 px-2.5 py-0.5 rounded-full">
                      Camera Off
                    </span>
                  )}
                </div>

                <div className="relative aspect-video overflow-hidden rounded-xl bg-gray-900 border border-gray-200 shadow-inner">
                  {cameraStreamActive ? (
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="h-full w-full object-cover -scale-x-100"
                    />
                  ) : (
                    <div className="flex h-full flex-col items-center justify-center p-4 text-center">
                      <VideoOff className="h-8 w-8 text-gray-400 mb-2" />
                      <p className="text-xs text-gray-300">Camera is off</p>
                      <p className="text-[11px] text-gray-500 mt-1">Turn on camera to register your face</p>
                    </div>
                  )}
                </div>

                {/* Face Registration Status Banner */}
                {faceRegisterMessage && (
                  <div className={`mt-3 rounded-lg p-2.5 text-xs ${
                    proctorRegistered
                      ? 'bg-green-50 text-green-800 border border-green-200'
                      : 'bg-amber-50 text-amber-800 border border-amber-200'
                  }`}>
                    {faceRegisterMessage}
                  </div>
                )}

                <div className="mt-3 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={handleStartCamera}
                    disabled={cameraStreamActive}
                    className={`flex-1 inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-xs font-medium transition-colors ${
                      cameraStreamActive
                        ? 'bg-gray-100 text-gray-400 cursor-not-allowed border border-gray-200'
                        : 'bg-gray-900 text-white hover:bg-gray-800 shadow-sm'
                    }`}
                  >
                    <Camera className="h-3.5 w-3.5" />
                    Turn On Camera
                  </button>

                  <button
                    type="button"
                    onClick={handleStopCamera}
                    disabled={!cameraStreamActive}
                    className={`inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-xs font-medium transition-colors ${
                      !cameraStreamActive
                        ? 'bg-gray-100 text-gray-400 cursor-not-allowed border border-gray-200'
                        : 'border border-error-200 bg-white text-error-600 hover:bg-error-50'
                    }`}
                  >
                    <CameraOff className="h-3.5 w-3.5" />
                    Turn Off
                  </button>

                  <button
                    type="button"
                    onClick={() => registerCandidateFace()}
                    disabled={!cameraStreamActive || registeringFace}
                    className={`w-full inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-2.5 text-xs font-semibold transition-colors ${
                      proctorRegistered
                        ? 'bg-green-600 text-white hover:bg-green-700'
                        : 'bg-primary-600 text-white hover:bg-primary-700 shadow-sm'
                    } disabled:opacity-50`}
                  >
                    {registeringFace ? (
                      <>
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        Scanning & Registering Face...
                      </>
                    ) : proctorRegistered ? (
                      <>
                        <CheckCircle2 className="h-3.5 w-3.5" />
                        Re-register Face
                      </>
                    ) : (
                      <>
                        <ShieldCheck className="h-3.5 w-3.5" />
                        Register My Face
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Microphone Check Card */}
              <div className="card p-5">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Mic className="h-4 w-4 text-primary-600" />
                    <span className="text-sm font-semibold text-gray-900">Microphone Check</span>
                  </div>
                  {micTestText ? (
                    <span className="inline-flex items-center gap-1 text-xs font-medium text-green-700 bg-green-50 px-2.5 py-0.5 rounded-full border border-green-200">
                      <CheckCircle2 className="h-3 w-3" /> Voice Detected
                    </span>
                  ) : micTesting ? (
                    <span className="inline-flex items-center gap-1 text-xs font-medium text-primary-700 bg-primary-50 px-2.5 py-0.5 rounded-full border border-primary-200">
                      <Loader2 className="h-3 w-3 animate-spin" /> Listening...
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-xs font-medium text-gray-500 bg-gray-100 px-2.5 py-0.5 rounded-full">
                      Ready to test
                    </span>
                  )}
                </div>

                <div className="rounded-xl border border-gray-200 bg-gray-50 p-3 text-xs min-h-[52px] flex items-center justify-center text-center">
                  {micTestText ? (
                    <p className="text-gray-900 font-medium italic">&quot;{micTestText}&quot;</p>
                  ) : micTesting ? (
                    <p className="text-primary-600 font-medium animate-pulse">Speak a few words into your microphone...</p>
                  ) : (
                    <p className="text-gray-400">Click &quot;Test Microphone&quot; and speak aloud</p>
                  )}
                </div>

                <div className="mt-3">
                  <button
                    type="button"
                    onClick={handleQuickMicTest}
                    className="w-full inline-flex items-center justify-center gap-1.5 rounded-lg border border-primary-200 bg-primary-50 px-3 py-2 text-xs font-medium text-primary-700 hover:bg-primary-100 transition-colors"
                  >
                    <Volume2 className="h-3.5 w-3.5" />
                    {micTesting ? 'Stop Mic Test' : 'Test Microphone'}
                  </button>
                </div>
              </div>
            </div>

            {/* Right: Interview Rules & Start Button */}
            <div className="space-y-4 flex flex-col justify-between">
              <div className="card p-5 space-y-4">
                <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-700">
                  Interview Rules & Guidelines
                </h2>

                <div className="space-y-3 text-xs text-gray-600 leading-relaxed">
                  <div className="flex items-start gap-2.5">
                    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-blue-100 text-[10px] font-bold text-blue-700">1</span>
                    <p><strong>Face Registration Required:</strong> Register your face using the camera before starting. During the interview, you will be verified continuously.</p>
                  </div>
                  <div className="flex items-start gap-2.5">
                    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-blue-100 text-[10px] font-bold text-blue-700">2</span>
                    <p><strong>Proctor Limits:</strong> Maximum <strong>3 warnings</strong> for wrong/unregistered face, <strong>5 warnings</strong> for no face detected, and <strong>3 warnings</strong> for multiple faces.</p>
                  </div>
                  <div className="flex items-start gap-2.5">
                    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-blue-100 text-[10px] font-bold text-blue-700">3</span>
                    <p><strong>Fullscreen & Screen Switching:</strong> Stay in fullscreen mode. Switching screens (Alt+Tab) or tabs has 1 warning allowed; a 2nd exit immediately terminates the interview.</p>
                  </div>
                  <div className="flex items-start gap-2.5">
                    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-blue-100 text-[10px] font-bold text-blue-700">4</span>
                    <p><strong>Zero-Loss Speech Recognition:</strong> Powered by <strong>Groq Whisper Large-v3-turbo</strong> with local fallback. Speaks continuously without dropping words; you can also edit or type your answers directly.</p>
                  </div>
                </div>
              </div>

              <div className="card p-5 bg-gradient-to-br from-primary-50 via-white to-primary-50 border border-primary-100">
                <h3 className="text-sm font-semibold text-gray-900 mb-1">
                  Ready to Start?
                </h3>
                <p className="text-xs text-gray-600 mb-4">
                  Starting the interview will activate Question 1 speech synthesis and begin timer recording.
                </p>

                <button
                  type="button"
                  onClick={handleStartInterview}
                  className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-primary-600 px-6 py-3.5 text-sm font-semibold text-white shadow-md transition hover:bg-primary-700 active:scale-[0.99]"
                >
                  Start Interview
                  <ArrowRight className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* Live Interview: Optimized UI Layout */
        /* Left Side: Question, Answer & Transcript | Right Side: User Video (Top), AI Avatar (Middle), Analytics (Bottom) */
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          {/* Eye-catching Blinking Proctoring Warning Banner */}
          {activeProctorWarning && (
            <div className="mb-6 animate-pulse">
              <div
                className={`flex flex-col sm:flex-row sm:items-center justify-between gap-3 rounded-2xl p-4 sm:p-5 text-white shadow-2xl ring-4 transition-all duration-300 ${
                  activeProctorWarning.level === 'critical'
                    ? 'bg-gradient-to-r from-red-600 via-rose-600 to-red-700 ring-red-400 shadow-red-500/50'
                    : 'bg-gradient-to-r from-amber-600 via-orange-600 to-amber-700 ring-amber-300 shadow-amber-500/50'
                }`}
              >
                <div className="flex items-start sm:items-center gap-3">
                  <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-white/20 backdrop-blur-sm animate-bounce">
                    <AlertTriangle className="h-6 w-6 text-white" />
                  </span>
                  <div>
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-[11px] font-black uppercase tracking-wider bg-white/25 px-2.5 py-0.5 rounded-full backdrop-blur-sm">
                        Proctoring Alert
                      </span>
                      <h4 className="text-sm sm:text-base font-black tracking-wide">
                        {activeProctorWarning.title}
                      </h4>
                    </div>
                    <p className="mt-1 text-xs sm:text-sm text-white/95 font-medium leading-relaxed">
                      {activeProctorWarning.message}
                    </p>
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-2 self-end sm:self-center">
                  <div className="flex flex-col items-center justify-center rounded-xl bg-black/30 px-3.5 py-1.5 border border-white/20">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-white/80">Strike</span>
                    <span className="text-base sm:text-lg font-black text-white">
                      {activeProctorWarning.count} / {activeProctorWarning.max}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left Column (lg:col-span-7 xl:col-span-8): Questions & What Is Being Answered */}
            <div className="lg:col-span-7 xl:col-span-8 space-y-6">
              {/* Question Card */}
              <div
                className="card p-6 sm:p-8 animate-slide-up shadow-sm border border-gray-200"
                key={currentIndex}
              >
                <div className="mb-3 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="flex h-7 w-7 items-center justify-center rounded-full bg-primary-100 text-xs font-bold text-primary-700">
                      {currentIndex + 1}
                    </span>

                    <span className="text-xs font-medium uppercase tracking-wide text-gray-400">
                      {currentQ.category?.replace('_', ' ') ||
                        currentQ.topic ||
                        'general'}
                    </span>
                  </div>

                  <span className="text-xs text-gray-400">
                    Question {currentIndex + 1} of {questions.length}
                  </span>
                </div>

                <h2
                  className="select-none text-xl font-semibold leading-relaxed text-gray-900 sm:text-2xl"
                  onCopy={(e) => e.preventDefault()}
                  onCut={(e) => e.preventDefault()}
                  onContextMenu={(e) => e.preventDefault()}
                >
                  {currentQ.question_text}
                </h2>

                {/* Voice speech-to-text transcript banner */}
                {(voicePhase === 'listening' || transcript || interimTranscript || isWhisperTranscribing) && (
                  <div className="mt-5 rounded-xl border border-primary-100 bg-primary-50/50 p-3.5">
                    <div className="mb-1.5 flex items-center gap-2">
                      <Mic className="h-3.5 w-3.5 text-primary-600" />
                      <span className="text-[11px] font-semibold uppercase tracking-wide text-primary-700">
                        Groq Whisper AI • Live Speech-to-Text
                      </span>
                      {isWhisperTranscribing ? (
                        <span className="ml-auto inline-flex items-center gap-1 text-[11px] font-medium text-amber-600">
                          <Loader2 className="h-3 w-3 animate-spin text-amber-600" />
                          Transcribing...
                        </span>
                      ) : voicePhase === 'listening' ? (
                        <span className="ml-auto inline-flex items-center gap-1 text-[11px] font-medium text-emerald-600">
                          <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                          Continuous Mic Active
                        </span>
                      ) : null}
                    </div>

                    <p className="min-h-[36px] whitespace-pre-wrap text-xs leading-relaxed text-gray-700">
                      {transcript}
                      {interimTranscript && (
                        <span className="italic text-gray-400">
                          {' '}
                          {interimTranscript}
                        </span>
                      )}
                      {!transcript && !interimTranscript && (
                        <span className="text-gray-400 italic">
                          Speak aloud into your microphone or type your answer below...
                        </span>
                      )}
                    </p>
                  </div>
                )}

                {/* Answer Textarea */}
                <div className="mt-5">
                  <label
                    className="label-text flex items-center justify-between"
                    htmlFor="answer"
                  >
                    <span>Your Answer</span>
                    <span className="text-[11px] font-normal text-gray-400">
                      Auto-saves automatically
                    </span>
                  </label>

                  <textarea
                    id="answer"
                    value={answer}
                    onChange={(e) => {
                      const val = e.target.value;
                      setAnswer(val);
                      answerRef.current = val;
                      finalChunksRef.current = val.trim() ? [val.trim()] : [];
                      lastFinalResultIndexRef.current = -1;
                      transcriptRef.current = val.trim();
                      sessionInterimRef.current = '';
                      setInterimTranscript('');
                    }}
                    placeholder="Type or speak your answer here. Be as detailed and comprehensive as you can."
                    className="input-field min-h-[220px] resize-y leading-relaxed text-sm"
                    disabled={ending}
                  />

                  <p className="helper-text mt-1.5 flex items-center justify-between">
                    <span>
                      {answer.trim().length > 0
                        ? `${answer.trim().split(/\s+/).length} words`
                        : 'Take your time. You can navigate between questions freely.'}
                    </span>
                  </p>
                </div>
              </div>

              {/* Error Banner */}
              {error && (
                <div className="flex items-start gap-2 rounded-lg bg-error-50 px-3 py-2.5 text-sm text-error-700">
                  <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              {/* Question Navigation Toolbar */}
              <div className="flex items-center justify-between pt-2">
                <button
                  onClick={() => goToQuestion(currentIndex - 1)}
                  disabled={currentIndex === 0 || saving || ending}
                  className="btn-secondary"
                >
                  <ArrowLeft className="h-4 w-4" />
                  Previous
                </button>

                <div className="flex items-center gap-1.5">
                  {questions.map((question, i) => (
                    <button
                      key={question.id}
                      onClick={() => goToQuestion(i)}
                      disabled={saving || ending}
                      className={`h-2 rounded-full transition-all ${
                        i === currentIndex
                          ? 'w-6 bg-primary-600'
                          : question.answer_text
                          ? 'w-2 bg-accent-500'
                          : 'w-2 bg-gray-300'
                      }`}
                      aria-label={`Go to question ${i + 1}`}
                    />
                  ))}
                </div>

                {currentIndex === questions.length - 1 ? (
                  <button
                    onClick={() => {
                      window.speechSynthesis?.cancel();
                      stopListening();
                      setShowEndConfirm(true);
                    }}
                    disabled={saving || ending}
                    className="btn-primary"
                  >
                    Finish & Review
                    <ArrowRight className="h-4 w-4" />
                  </button>
                ) : (
                  <button
                    onClick={() => goToQuestion(currentIndex + 1)}
                    disabled={saving || ending}
                    className="btn-primary"
                  >
                    {saving && (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    )}
                    Next
                    <ArrowRight className="h-4 w-4" />
                  </button>
                )}
              </div>
            </div>

            {/* Right Column (lg:col-span-5 xl:col-span-4): User Video (Top), AI (Middle), Analytics (Bottom) */}
            <div className="lg:col-span-5 xl:col-span-4 space-y-4">
              {/* 1. RIGHT TOP: Video of User with Camera/Mic Controls */}
              {cameraEnabled && (
                <div className="card p-3 shadow-sm border border-gray-200">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-semibold text-gray-700 uppercase tracking-wide">
                      Candidate Feed
                    </span>

                    {/* Face Identified / Not Identified Proctor Badge */}
                    {cameraStreamActive && (
                      activeProctorWarning ? (
                        <span className="animate-pulse inline-flex items-center gap-1.5 rounded-full bg-red-600 px-2.5 py-0.5 text-[11px] font-bold text-white shadow-md">
                          <span className="h-2 w-2 rounded-full bg-white animate-ping" />
                          Alert ({activeProctorWarning.count}/{activeProctorWarning.max})
                        </span>
                      ) : (
                        <span
                          className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${
                            !proctorResult
                              ? 'bg-gray-100 text-gray-600'
                              : proctorResult.status === 'VERIFIED'
                              ? 'bg-green-50 text-green-700 border border-green-200'
                              : 'bg-red-50 text-red-700 border border-red-200'
                          }`}
                        >
                          <span
                            className={`h-2 w-2 rounded-full ${
                              !proctorResult
                                ? 'bg-gray-400'
                                : proctorResult.status === 'VERIFIED'
                                ? 'bg-green-500'
                                : 'bg-red-500'
                            }`}
                          />
                          {!proctorResult
                            ? 'Checking Face...'
                            : proctorResult.status === 'VERIFIED'
                            ? 'Face Identified ✓'
                            : 'Face Not Identified ⚠️'}
                        </span>
                      )
                    )}
                  </div>

                  <div className={`relative aspect-video overflow-hidden rounded-xl border transition-all duration-300 ${
                    activeProctorWarning
                      ? 'ring-4 ring-red-500 ring-offset-2 animate-pulse border-red-500 shadow-2xl shadow-red-500/50'
                      : 'border-gray-200 bg-gray-900 shadow-inner'
                  }`}>
                    {activeProctorWarning && (
                      <div className="absolute top-2 left-2 z-20 animate-pulse flex items-center gap-1.5 rounded-full bg-red-600 px-2.5 py-1 text-[10px] font-black uppercase tracking-wider text-white shadow-xl ring-2 ring-white/60">
                        <span className="h-2 w-2 rounded-full bg-white animate-ping" />
                        STRIKE {activeProctorWarning.count} OF {activeProctorWarning.max}
                      </div>
                    )}
                    {cameraPermission === 'denied' ? (
                      <div className="flex h-full flex-col items-center justify-center p-3 text-center">
                        <VideoOff className="mb-1.5 h-6 w-6 text-gray-400" />
                        <p className="text-xs text-gray-300">Camera access denied</p>
                      </div>
                    ) : cameraPermission === 'loading' ? (
                      <div className="flex h-full items-center justify-center">
                        <Loader2 className="mr-2 h-4 w-4 animate-spin text-white" />
                        <span className="text-xs text-white/70">Starting...</span>
                      </div>
                    ) : cameraStreamActive ? (
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="h-full w-full object-cover -scale-x-100"
                      />
                    ) : (
                      <div className="flex h-full flex-col items-center justify-center p-3 text-center">
                        <VideoOff className="h-6 w-6 text-white/40 mb-1" />
                        <span className="text-xs text-white/60">Camera turned off</span>
                      </div>
                    )}

                    <div className="absolute bottom-2 left-2 rounded-md bg-black/60 px-2 py-0.5 backdrop-blur-sm">
                      <span className="text-[10px] font-medium text-white">You</span>
                    </div>
                  </div>

                  {/* Camera & Mic Controls Toolbar */}
                  <div className="mt-2.5 flex flex-wrap items-center justify-between gap-1.5 pt-2 border-t border-gray-100">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <button
                        type="button"
                        onClick={handleStartCamera}
                        disabled={cameraStreamActive || ending}
                        className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
                          cameraStreamActive
                            ? 'bg-gray-100 text-gray-400 cursor-not-allowed border border-gray-200'
                            : 'bg-gray-900 text-white hover:bg-gray-800'
                        }`}
                      >
                        <Camera className="h-3 w-3" />
                        Start Cam
                      </button>

                      <button
                        type="button"
                        onClick={handleStopCamera}
                        disabled={!cameraStreamActive || ending}
                        className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
                          !cameraStreamActive
                            ? 'bg-gray-100 text-gray-400 cursor-not-allowed border border-gray-200'
                            : 'border border-error-300 bg-white text-error-600 hover:bg-error-50'
                        }`}
                      >
                        <CameraOff className="h-3 w-3" />
                        Stop Cam
                      </button>

                      <button
                        type="button"
                        onClick={handleStartMic}
                        disabled={micEnabled || ending}
                        className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
                          micEnabled
                            ? 'bg-gray-100 text-gray-400 cursor-not-allowed border border-gray-200'
                            : 'bg-primary-600 text-white hover:bg-primary-700'
                        }`}
                      >
                        <Mic className="h-3 w-3" />
                        Mic On
                      </button>

                      <button
                        type="button"
                        onClick={handleStopMic}
                        disabled={!micEnabled || ending}
                        className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
                          !micEnabled
                            ? 'bg-gray-100 text-gray-400 cursor-not-allowed border border-gray-200'
                            : 'border border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
                        }`}
                      >
                        <MicOff className="h-3 w-3" />
                        Mic Off
                      </button>
                    </div>

                    {cameraDevices.length > 1 && (
                      <button
                        type="button"
                        onClick={handleCameraDeviceSwitch}
                        disabled={ending}
                        className="inline-flex items-center gap-1 rounded-md border border-gray-300 bg-white px-2 py-1 text-[11px] font-medium text-gray-700 hover:bg-gray-50"
                      >
                        <RefreshCw className="h-3 w-3" />
                        Switch
                      </button>
                    )}
                  </div>
                </div>
              )}

              {/* 2. RIGHT: Live Analytics & Emotion Signals */}
              <div className="card p-3.5 shadow-sm border border-gray-200">
                <div className="flex items-center gap-2 mb-2.5 pb-2 border-b border-gray-100">
                  <Activity className="h-3.5 w-3.5 text-primary-600" />
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-700">
                    Live Analytics & Emotion Signals
                  </h3>
                </div>

                <div className="space-y-2">
                  {/* Facial Emotion */}
                  <div className="rounded-lg bg-gray-50 border border-gray-200/80 p-2">
                    <div className="flex items-center justify-between text-xs mb-0.5">
                      <span className="font-semibold text-gray-500 uppercase tracking-wider text-[10px]">
                        Facial Emotion
                      </span>
                      {typeof faceEmotion?.confidence === 'number' && (
                        <span className="text-gray-500 font-mono text-[10px]">
                          {(faceEmotion.confidence * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium capitalize text-gray-900">
                        {faceEmotion?.emotion || (cameraStreamActive ? 'Analyzing...' : 'Camera Off')}
                      </span>
                      {faceEmotion?.emotion && (
                        <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                      )}
                    </div>
                  </div>

                  {/* Voice Emotion */}
                  <div className="rounded-lg bg-gray-50 border border-gray-200/80 p-2">
                    <div className="flex items-center justify-between text-xs mb-0.5">
                      <span className="font-semibold text-gray-500 uppercase tracking-wider text-[10px]">
                        Voice Emotion
                      </span>
                      {typeof voiceEmotion?.confidence === 'number' && (
                        <span className="text-gray-500 font-mono text-[10px]">
                          {(voiceEmotion.confidence * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium capitalize text-gray-900">
                        {voiceEmotion?.emotion || (micEnabled ? 'Listening...' : 'Mic Off')}
                      </span>
                      {voiceEmotion?.emotion && (
                        <span className="h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
                      )}
                    </div>
                  </div>

                  {/* Text Emotion */}
                  <div className="rounded-lg bg-gray-50 border border-gray-200/80 p-2">
                    <div className="flex items-center justify-between text-xs mb-0.5">
                      <span className="font-semibold text-gray-500 uppercase tracking-wider text-[10px]">
                        Text Emotion
                      </span>
                      {typeof textEmotion?.confidence === 'number' && (
                        <span className="text-gray-500 font-mono text-[10px]">
                          {(textEmotion.confidence * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium capitalize text-gray-900">
                        {textEmotion?.emotion || (answer.trim() ? 'Evaluating...' : 'Waiting for answer')}
                      </span>
                      {textEmotion?.emotion && (
                        <span className="h-2 w-2 rounded-full bg-emerald-500" />
                      )}
                    </div>
                  </div>

                  {/* Multimodal Fusion Summary */}
                  {fusionResult?.success && (
                    <div className="rounded-lg bg-primary-50/50 border border-primary-100 p-2 mt-2">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-semibold text-primary-900 text-[10px] uppercase tracking-wider">
                          Multimodal Fusion
                        </span>
                        {typeof fusionResult.overall_score === 'number' && (
                          <span className="text-xs font-bold text-primary-700">
                            Score: {fusionResult.overall_score.toFixed(1)}
                          </span>
                        )}
                      </div>
                      {fusionResult.dominant_emotion && (
                        <p className="text-[11px] text-gray-600 mb-1.5">
                          Dominant: <strong className="capitalize text-gray-900">{fusionResult.dominant_emotion}</strong>
                        </p>
                      )}
                      {fusionResult.behavioral_metrics && (
                        <div className="grid grid-cols-2 gap-1 text-[10px]">
                          <div className="bg-white/80 rounded px-1.5 py-0.5 flex justify-between">
                            <span className="text-gray-500">Confidence</span>
                            <span className="font-semibold">{fusionResult.behavioral_metrics.confidence.toFixed(1)}</span>
                          </div>
                          <div className="bg-white/80 rounded px-1.5 py-0.5 flex justify-between">
                            <span className="text-gray-500">Fluency</span>
                            <span className="font-semibold">{fusionResult.behavioral_metrics.fluency.toFixed(1)}</span>
                          </div>
                          <div className="bg-white/80 rounded px-1.5 py-0.5 flex justify-between">
                            <span className="text-gray-500">Composure</span>
                            <span className="font-semibold">{fusionResult.behavioral_metrics.composure.toFixed(1)}</span>
                          </div>
                          <div className="bg-white/80 rounded px-1.5 py-0.5 flex justify-between">
                            <span className="text-gray-500">Stress</span>
                            <span className="font-semibold">{fusionResult.behavioral_metrics.stress.toFixed(1)}</span>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* End confirmation modal */}
      {showEndConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 animate-fade-in">
          <div className="card max-w-md p-6 animate-slide-up">
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-error-50">
                <X className="h-5 w-5 text-error-600" />
              </div>

              <div>
                <h3 className="text-lg font-semibold text-gray-900">
                  End interview?
                </h3>

                <p className="mt-1 text-sm text-gray-600">
                  Your answers will be saved and the AI will
                  evaluate your interview. You'll then see your
                  feedback summary.
                </p>
              </div>
            </div>

            <div className="mt-6 flex justify-end gap-2">
              <button
                onClick={() => setShowEndConfirm(false)}
                disabled={ending}
                className="btn-secondary"
              >
                Keep practicing
              </button>

              <button
                onClick={handleEnd}
                disabled={ending}
                className="btn-primary bg-error-600 hover:bg-error-700"
              >
                {ending ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    AI evaluating...
                  </>
                ) : (
                  'End & view results'
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

