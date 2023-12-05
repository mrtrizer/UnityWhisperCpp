using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Abuksigun.WhisperCpp
{
    public sealed class WhisperModel : IDisposable
    {
        public class WhisperException : Exception
        {
            public WhisperException(string message) : base(message) { }
        }

        public class WhisperParams
        {
            public int NThreads { get; set; }
            public int StepMs { get; set; } = 3000;
            public int LengthMs { get; set; } = 10000;
            public int KeepMs { get; set; } = 200;
            public int CaptureId { get; set; } = -1;
            public int MaxTokens { get; set; } = 32;
            public int AudioCtx { get; set; } = 0;

            public float VadThold { get; set; } = 0.6f;
            public float FreqThold { get; set; } = 100.0f;

            public bool SpeedUp { get; set; } = false;
            public bool Translate { get; set; } = false;
            public bool NoFallback { get; set; } = false;
            public bool PrintSpecial { get; set; } = false;
            public bool NoContext { get; set; } = true;
            public bool NoTimestamps { get; set; } = false;
            public bool TinyDiarize { get; set; } = false;
            public bool UseGpu { get; set; } = true;

            public string Language { get; set; }

            public WhisperParams(string language, int nThreads)
            {
                Language = language;
                NThreads = nThreads;
            }
        }

        public const int SampleRate = 16000;

        IntPtr contextPointer;
        readonly CancellationTokenSource disposeCancellationTokenSource = new();
        readonly WhisperParams whisperParams;

        readonly Queue<float[]> pcmf32Buffers = new();
        volatile bool running = false;

        public unsafe WhisperLibrary.WhisperContext* NativeContextPointer => (WhisperLibrary.WhisperContext*)contextPointer;

        public static async Task<WhisperModel> LoadModel(string modelPath, WhisperParams whisperParams, bool useGpu)
        {
            return await Task.Run(() => {
                var cparams = new WhisperLibrary.WhisperContextParams { use_gpu = useGpu };

                unsafe
                {
                    var contextPointer = WhisperLibrary.whisper_init_from_file_with_params(modelPath, cparams);
                    if (contextPointer == IntPtr.Zero)
                    {
                        throw new WhisperException("Failed to initialize Whisper context.");
                    }

                    return new WhisperModel(contextPointer, whisperParams);
                }
            });
        }

        WhisperModel(IntPtr contextPointer, WhisperParams whisperParams)
        {
            this.contextPointer = contextPointer;
            this.whisperParams = whisperParams;
        }

        ~WhisperModel()
        {
            Dispose();
        }

        public unsafe void Dispose()
        {
            if (contextPointer == IntPtr.Zero)
                return;
            disposeCancellationTokenSource.Cancel();

            WhisperLibrary.whisper_free(NativeContextPointer);
            contextPointer = IntPtr.Zero;
        }

        public void AddPcmf32(float[] pcmf32)
        {
            if (!running)
                return;
            lock (pcmf32Buffers)
                pcmf32Buffers.Enqueue(pcmf32);
        }

        public string AudioToText(float[] pcmf32, List<int> contextTokens = null)
        {
            WhisperLibrary.WhisperFullParams wparams = WhisperLibrary.whisper_full_default_params(WhisperLibrary.WhisperSamplingStrategy.WHISPER_SAMPLING_GREEDY);

            wparams.print_progress = true;
            wparams.print_special = whisperParams.PrintSpecial;
            wparams.print_realtime = true;
            wparams.print_timestamps = !whisperParams.NoTimestamps;
            wparams.translate = whisperParams.Translate;
            wparams.single_segment = true;
            wparams.max_tokens = whisperParams.MaxTokens;
            wparams.duration_ms = pcmf32.Length * 1000 / SampleRate;
            unsafe
            {
                int[] contextTokensArray = contextTokens?.ToArray() ?? Array.Empty<int>();
                byte[] languageAscii = Encoding.ASCII.GetBytes(whisperParams.Language + '\0');
                fixed (byte* languagePtr = languageAscii)
                fixed (int* contextTokensArrayPtr = contextTokensArray)
                {
                    wparams.language = languagePtr;
                    wparams.n_threads = whisperParams.NThreads;

                    wparams.audio_ctx = whisperParams.AudioCtx;
                    wparams.speed_up = whisperParams.SpeedUp;

                    wparams.tdrz_enable = whisperParams.TinyDiarize;
                    wparams.temperature_inc = whisperParams.NoFallback ? 0.0f : wparams.temperature_inc;

                    wparams.prompt_tokens = contextTokensArrayPtr;
                    wparams.prompt_n_tokens = contextTokensArray.Length;
                    fixed (float* pcmf32Ptr = pcmf32)
                    {
                        int result = WhisperLibrary.whisper_full(NativeContextPointer, wparams, pcmf32Ptr, pcmf32.Length);
                        if (result != 0)
                            throw new WhisperException($"Failed to process audio. Code: {result}");
                    }
                }
                int nSegments = WhisperLibrary.whisper_full_n_segments(NativeContextPointer);
                if (nSegments == 0)
                    return null;
                StringBuilder stringBuilder = new StringBuilder();
                for (int i = 0; i < nSegments; ++i)
                {
                    string text = WhisperLibrary.whisper_full_get_segment_text(NativeContextPointer, i);
                    stringBuilder.Append(text);
                }
                return stringBuilder.ToString();
            }
        }

        public Task<string> AudioToTextAsync(float[] pcmf32)
        {
            return Task.Run(() => AudioToText(pcmf32));
        }

        void Run(Action<string> textCallback, CancellationToken ct)
        {
            if (running)
                throw new WhisperException("WhisperModel is already running.");
            running = true;

            var contextTokens = new List<int>(whisperParams.MaxTokens);
            while (!ct.IsCancellationRequested)
            {
                float[] pcmf32;
                lock (pcmf32Buffers)
                {
                    if (pcmf32Buffers.Count == 0)
                    {
                        try
                        {
                            Task.Delay(10, ct).Wait();
                        }
                        catch
                        {
                            return;
                        }
                        continue;
                    }
                    pcmf32 = pcmf32Buffers.Dequeue();
                }

                string text = AudioToText(pcmf32, contextTokens);
                if (text != null)
                    textCallback(text);

                unsafe
                {
                    int segmentsLength = WhisperLibrary.whisper_full_n_segments(NativeContextPointer);
                    for (int i = 0; i < segmentsLength; ++i)
                    {
                        int tokensLength = WhisperLibrary.whisper_full_n_tokens(NativeContextPointer, i);
                        for (int j = 0; j < tokensLength; ++j)
                            contextTokens.Add(WhisperLibrary.whisper_full_get_token_id(NativeContextPointer, i, j));
                    }
                }
            }
            lock(pcmf32Buffers)
                pcmf32Buffers.Clear();
            running = false;
        }

        public Task RunAsync(Action<string> textCallback, CancellationToken? ct)
        {
            var tokenSource = ct.HasValue ? CancellationTokenSource.CreateLinkedTokenSource(disposeCancellationTokenSource.Token, ct.Value) : disposeCancellationTokenSource;
            return Task.Run(() => Run(textCallback, tokenSource.Token));
        }


        public static bool VadSimple(List<float> pcmf32, int sampleRate, int lastMs, float vadThold, float freqThold)
        {
            int nSamples = pcmf32.Count;
            int nSamplesLast = (sampleRate * lastMs) / 1000;

            if (nSamplesLast >= nSamples)
            {
                // not enough samples - assume no speech
                return false;
            }

            if (freqThold > 0.0f)
            {
                HighPassFilter(pcmf32, freqThold, sampleRate);
            }

            float energyAll = 0.0f;
            float energyLast = 0.0f;

            for (int i = 0; i < nSamples; i++)
            {
                energyAll += Math.Abs(pcmf32[i]);

                if (i >= nSamples - nSamplesLast)
                {
                    energyLast += Math.Abs(pcmf32[i]);
                }
            }

            energyAll /= nSamples;
            energyLast /= nSamplesLast;

            Debug.Log($"VadSimple: energy_all: {energyAll}, energy_last: {energyLast}, vad_thold: {vadThold}, freq_thold: {freqThold}");

            return energyLast <= vadThold * energyAll;
        }

        public static void HighPassFilter(List<float> data, float cutoff, float sampleRate)
        {
            const float pi = (float)Math.PI;
            float rc = 1.0f / (2.0f * pi * cutoff);
            float dt = 1.0f / sampleRate;
            float alpha = dt / (rc + dt);

            float y = data[0];

            for (int i = 1; i < data.Count; i++)
            {
                y = alpha * (y + data[i] - data[i - 1]);
                data[i] = y;
            }
        }
    }
}
