using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Abuksigun.WhisperCpp
{
    [ExecuteInEditMode]
    public class WhisperExample : MonoBehaviour
    {
        CancellationTokenSource cts;
        WhisperModel model;

        // Download model here: https://huggingface.co/TheBloke/speechless-mistral-dolphin-orca-platypus-samantha-7B-GGUF/blob/main/speechless-mistral-dolphin-orca-platypus-samantha-7b.Q4_K_M.gguf
        [SerializeField] string modelPath = "Models/ggml-medium.bin"; // Download model from provided URL
        [SerializeField] int sampleRate = 16000;
        [SerializeField] int lengthSeconds = 1;
        [SerializeField] AudioSource audioSource;
        [SerializeField] int micDeviceIndex = -1; // Run ListMicrophones to see available devices with indices

        [ContextMenu("List Microphones")]
        public void ListMicrophones()
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < Microphone.devices.Length; i++)
            {
                sb.AppendLine($"{i}: {Microphone.devices[i]}");
            }
            Debug.Log(sb.ToString());
        }

        [ContextMenu("Run")]
        public async void RunAsync()
        {
            if (micDeviceIndex < 0 || micDeviceIndex >= Microphone.devices.Length)
            {
                Debug.LogError($"Invalid mic device index: {micDeviceIndex}");
                return;
            }

            int threadsN = SystemInfo.processorCount;
            var whisperParams = new WhisperModel.WhisperParams("en", threadsN);

            long startTicks = DateTime.Now.Ticks;
            string fullModelPath = Path.Join(Application.streamingAssetsPath, modelPath);
            model ??= await WhisperModel.LoadModel(fullModelPath, whisperParams, true);

            long loadTime = DateTime.Now.Ticks - startTicks;
            if (loadTime > 0)
                Debug.Log($"Model loaded in { loadTime / TimeSpan.TicksPerSecond } seconds.");

            cts = new CancellationTokenSource();

            string micDevice = Microphone.devices[micDeviceIndex];
            audioSource.clip = Microphone.Start(micDevice, true, lengthSeconds, sampleRate);
            audioSource.Play();

            model?.RunAsync(text => Debug.Log(text), cts.Token);

            try
            {
                while (!cts.IsCancellationRequested)
                {
                    await Task.Delay(lengthSeconds * 1000); // Wait for audio clip to fill

                    Debug.Log(Microphone.GetPosition(micDevice));

                    float[] audioData = new float[audioSource.clip.samples * audioSource.clip.channels];
                    audioSource.clip.GetData(audioData, 0);

                    int deviationCount = audioData.Count(x => Mathf.Abs(x) > 0.1f);
                    if (deviationCount > 0)
                    {
                        Debug.Log($"Voice detected! {deviationCount}");
                        model.AddPcmf32(audioData);
                    }
                }
            }
            finally
            {
                Microphone.End(micDevice);
                audioSource.Stop();
            }
        }

        [ContextMenu("Stop")]
        public void Stop()
        {
            cts?.Cancel();
            Microphone.End(null);
        }
    }
}