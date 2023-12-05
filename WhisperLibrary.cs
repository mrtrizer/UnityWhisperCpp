using System;
using System.Runtime.InteropServices;

namespace Abuksigun.WhisperCpp
{
    public static unsafe class WhisperLibrary
    {
        //const string WhisperDll = "whisper"; // Name of the DLL
        const string WhisperDll = "whisper_debug"; // Name of the DLL

        [StructLayout(LayoutKind.Sequential)]   
        public struct WhisperContext
        {
            // reference only
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct WhisperContextParams
        {
            public bool use_gpu;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct WhisperTokenData
        {
            public int Id;    // Assuming whisper_token is an int type
            public int Tid;   // Assuming whisper_token is an int type

            public float P;    // Probability of the token
            public float Plog; // Log probability of the token
            public float Pt;   // Probability of the timestamp token
            public float Ptsum;// Sum of probabilities of all timestamp tokens

            // Token-level timestamp data
            public long T0;    // Start time of the token
            public long T1;    // End time of the token

            public float Vlen; // Voice length of the token
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void WhisperNewSegmentCallback(IntPtr ctx, IntPtr state, int n_new, IntPtr user_data);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void WhisperProgressCallback(IntPtr ctx, IntPtr state, int progress, IntPtr user_data);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate bool WhisperEncoderBeginCallback(IntPtr ctx, IntPtr state, IntPtr user_data);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate bool WhisperAbortCallback(IntPtr user_data);

        public enum WhisperSamplingStrategy
        {
            WHISPER_SAMPLING_GREEDY,      // similar to OpenAI's GreedyDecoder
            WHISPER_SAMPLING_BEAM_SEARCH, // similar to OpenAI's BeamSearchDecoder
        };

        public struct Greedy
        {
            public int best_of; // This corresponds to the best_of parameter in Whisper's greedy decoding method.
        }

        public struct BeamSearch
        {
            public int beam_size; // This corresponds to the beam_size parameter in Whisper's beam search decoding method.

            public float patience; // This is a placeholder for the 'patience' parameter as mentioned in the TODO comment.
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct WhisperFullParams
        {
            public WhisperSamplingStrategy strategy;

            public int n_threads;
            public int n_max_text_ctx;
            public int offset_ms;
            public int duration_ms;

            [MarshalAs(UnmanagedType.I1)] public bool translate;
            [MarshalAs(UnmanagedType.I1)] public bool no_context;
            [MarshalAs(UnmanagedType.I1)] public bool no_timestamps;
            [MarshalAs(UnmanagedType.I1)] public bool single_segment;
            [MarshalAs(UnmanagedType.I1)] public bool print_special;
            [MarshalAs(UnmanagedType.I1)] public bool print_progress;
            [MarshalAs(UnmanagedType.I1)] public bool print_realtime;
            [MarshalAs(UnmanagedType.I1)] public bool print_timestamps;

            [MarshalAs(UnmanagedType.I1)] public bool token_timestamps;
            public float thold_pt;
            public float thold_ptsum;
            public int max_len;
            [MarshalAs(UnmanagedType.I1)] public bool split_on_word;
            public int max_tokens;

            [MarshalAs(UnmanagedType.I1)] public bool speed_up;
            [MarshalAs(UnmanagedType.I1)] public bool debug_mode;
            public int audio_ctx;

            [MarshalAs(UnmanagedType.I1)] public bool tdrz_enable;

            public IntPtr initial_prompt;
            public int* prompt_tokens; // Should be a pointer to an array of whisper_token
            public int prompt_n_tokens;

            public byte* language;
            [MarshalAs(UnmanagedType.I1)] public bool detect_language;

            [MarshalAs(UnmanagedType.I1)] public bool suppress_blank;
            [MarshalAs(UnmanagedType.I1)] public bool suppress_non_speech_tokens;

            public float temperature;
            public float max_initial_ts;
            public float length_penalty;

            public float temperature_inc;
            public float entropy_thold;
            public float logprob_thold;
            public float no_speech_thold;

            public Greedy greedy;
            public BeamSearch beam_search;

            public IntPtr WhisperNewSegmentCallback;
            public IntPtr new_segment_callback_user_data;

            public IntPtr WhisperProgressCallback;
            public IntPtr progress_callback_user_data;

            public IntPtr WhisperEncoderBeginCallback;
            public IntPtr encoder_begin_callback_user_data;

            public WhisperAbortCallback abort_callback;
            public IntPtr abort_callback_user_data;

            public IntPtr logits_filter_callback;
            public IntPtr logits_filter_callback_user_data;

            public IntPtr grammar_rules;
            public int n_grammar_rules;
            public int i_start_rule;
            public float grammar_penalty;
        }


        [DllImport(WhisperDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool whisper_params_parse(int argc, string[] argv, WhisperContextParams* @params);

        [DllImport(WhisperDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int whisper_lang_id([MarshalAs(UnmanagedType.LPStr)] string lang);

        [DllImport(WhisperDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void whisper_print_usage(int argc, string[] argv, WhisperContextParams @params);

        [DllImport(WhisperDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr whisper_init_from_file_with_params([MarshalAs(UnmanagedType.LPStr)] string pathModel, WhisperContextParams @params);

        [DllImport(WhisperDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool whisper_is_multilingual(WhisperContext* ctx);

        [DllImport(WhisperDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int whisper_full(WhisperContext* ctx, WhisperFullParams @params, float* samples, int nSamples);

        [DllImport(WhisperDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern WhisperFullParams whisper_full_default_params(WhisperSamplingStrategy strategy);

        [DllImport(WhisperDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int whisper_full_n_segments(WhisperContext* ctx);

        [DllImport(WhisperDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern string whisper_full_get_segment_text(WhisperContext* ctx, int iSegment);

        [DllImport(WhisperDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern long whisper_full_get_segment_t0(WhisperContext* ctx, int iSegment);

        [DllImport(WhisperDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern long whisper_full_get_segment_t1(WhisperContext* ctx, int iSegment);

        [DllImport(WhisperDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool whisper_full_get_segment_speaker_turn_next(WhisperContext* ctx, int iSegment);

        [DllImport(WhisperDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int whisper_full_n_tokens(WhisperContext* ctx, int iSegment);

        [DllImport(WhisperDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int whisper_full_get_token_id(WhisperContext* ctx, int iSegment, int iToken);

        [DllImport(WhisperDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void whisper_print_timings(WhisperContext* ctx);

        [DllImport(WhisperDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void whisper_free(WhisperContext* ctx);
    }
}