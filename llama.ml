open Core

(* Configuration *)
module Config = struct
  type t = {
    dim : int;
    hidden_dim : int;
    n_layers : int;
    n_heads : int;
    n_kv_heads : int;
    vocab_size : int;
    seq_len : int;
  } [@@deriving sexp]
end

(* Transformer weights *)
module TransformerWeights = struct
  type t = {
    token_embedding_table : float array;
    rms_att_weight : float array array;
    rms_ffn_weight : float array array;
    wq : float array array;
    wk : float array array;
    wv : float array array;
    wo : float array array;
    w1 : float array array;
    w2 : float array array;
    w3 : float array array;
    rms_final_weight : float array;
    wcls : float array option;
  } [@@deriving sexp]
end

(* Run state *)
module RunState = struct
  type t = {
    x : float array;
    xb : float array;
    xb2 : float array;
    hb : float array;
    hb2 : float array;
    q : float array;
    k : float array;
    v : float array;
    att : float array;
    logits : float array;
    key_cache : float array array;
    value_cache : float array array;
  } [@@deriving sexp]

  let create config =
    let dim = config.Config.dim in
    let hidden_dim = config.Config.hidden_dim in
    let n_heads = config.Config.n_heads in
    let n_layers = config.Config.n_layers in
    let seq_len = config.Config.seq_len in
    let kv_dim = (dim * config.Config.n_kv_heads) / n_heads in
    {
      x = Array.create ~len:dim 0.0;
      xb = Array.create ~len:dim 0.0;
      xb2 = Array.create ~len:dim 0.0;
      hb = Array.create ~len:hidden_dim 0.0;
      hb2 = Array.create ~len:hidden_dim 0.0;
      q = Array.create ~len:dim 0.0;
      k = Array.create ~len:dim 0.0;
      v = Array.create ~len:dim 0.0;
      att = Array.create ~len:(n_heads * seq_len) 0.0;
      logits = Array.create ~len:config.Config.vocab_size 0.0;
      key_cache = Array.create ~len:(n_layers * seq_len * kv_dim) 0.0;
      value_cache = Array.create ~len:(n_layers * seq_len * kv_dim) 0.0;
    }
end

(* Transformer *)
module Transformer = struct
  type t = {
    config : Config.t;
    weights : TransformerWeights.t;
    state : RunState.t;
    fd : Unix.file_descr;
    data : float array;
    file_size : int;
  } [@@deriving sexp]

  let rmsnorm o x weight =
    let size = Array.length x in
    let ss = ref 0.0 in
    for j = 0 to size - 1 do
      ss := !ss +. (x.(j) *. x.(j));
    done;
    ss := !ss /. Float.of_int size;
    ss := !ss +. 1e-5;
    ss := 1.0 /. sqrt !ss;
    for j = 0 to size - 1 do
      o.(j) <- weight.(j) *. (!ss *. x.(j));
    done

  let softmax x =
    let size = Array.length x in
    let max_val = ref x.(0) in
    for i = 1 to size - 1 do
      if x.(i) > !max_val then max_val := x.(i);
    done;
    let sum = ref 0.0 in
    for i = 0 to size - 1 do
      x.(i) <- exp (x.(i) -. !max_val);
      sum := !sum +. x.(i);
    done;
    for i = 0 to size - 1 do
      x.(i) <- x.(i) /. !sum;
    done

  let matmul xout x w n d =
    let x_f = Array.map ~f:Float.of_int x in
    for i = 0 to d - 1 do
      let val_ = ref 0.0 in
      for j = 0 to n - 1 do
        val_ := !val_ +. (w.(i * n + j) *. x_f.(j));
      done;
      xout.(i) <- !val_;
    done

  let forward t token pos =
    let config = t.config in
    let weights = t.weights in
    let state = t.state in
    let dim = config.Config.dim in
    let kv_dim = (config.Config.dim * config.Config.n_kv_heads) / config.Config.n_heads in
    let hidden_dim = config.Config.hidden_dim in
    let head_size = dim / config.Config.n_heads in
    let content_row = Array.sub weights.TransformerWeights.token_embedding_table ~pos:(token * dim) ~len:dim in
    Array.blit ~src:content_row ~src_pos:0 ~dst:state.RunState.x ~dst_pos:0 ~len:dim;
    for l = 0 to config.Config.n_layers - 1 do
      let loff = l * config.Config.seq_len * kv_dim in
      rmsnorm state.RunState.xb state.RunState.x weights.TransformerWeights.rms_att_weight.(l);
      state.RunState.k <- Array.sub state.RunState.key_cache ~pos:(loff + pos * kv_dim) ~len:kv_dim;
      state.RunState.v <- Array.sub state.RunState.value_cache ~pos:(loff + pos * kv_dim) ~len:kv_dim;
      matmul state.RunState.q state.RunState.xb weights.TransformerWeights.wq.(l) dim dim;
      matmul state.RunState.k state.RunState.xb weights.TransformerWeights.wk.(l) dim kv_dim;
      matmul state.RunState.v state.RunState.xb weights.TransformerWeights.wv.(l) dim kv_dim;
      for i = 0 to dim - 1 do
        let head_dim = i mod head_size in
        let freq = 1.0 /. (10000.0 ** (Float.of_int head_dim /. Float.of_int head_size)) in
        let val_ = Float.of_int pos *. freq in
        let fcr = cos val_ in
        let fci = sin val_ in
        let rotn = if i < kv_dim then 2 else 1 in
        for v = 0 to rotn - 1 do
          let vec = if v = 0 then state.RunState.q else state.RunState.k in
          let v0 = vec.(i) in
          let v1 = vec.(i + 1) in
          vec.(i) <- v0 *. fcr -. v1 *. fci;
          vec.(i + 1) <- v0 *. fci +. v1 *. fcr;
        done;
      done;
      for h = 0 to config.Config.n_heads - 1 do
        let q = Array.sub state.RunState.q ~pos:(h * head_size) ~len:head_size in
        let att = Array.sub state.RunState.att ~pos:(h * config.Config.seq_len) ~len:config.Config.seq_len in
        for t = 0 to pos do
          let k = Array.sub state.RunState.key_cache ~pos:(loff + t * kv_dim + (h / (config.Config.n_heads / config.Config.n_kv_heads)) * head_size) ~len:head_size in
          let score = ref 0.0 in
          for i = 0 to head_size - 1 do
            score := !score +. (q.(i) *. k.(i));
          done;
          score := !score /. sqrt (Float.of_int head_size);
          att.(t) <- !score;
        done;
        softmax (Array.sub att ~pos:0 ~len:(pos + 1));
        let xb = Array.sub state.RunState.xb ~pos:(h * head_size) ~len:head_size in
        Array.fill xb ~pos:0 ~len:head_size 0.0;
        for t = 0 to pos do
          let v = Array.sub state.RunState.value_cache ~pos:(loff + t * kv_dim + (h / (config.Config.n_heads / config.Config.n_kv_heads)) * head_size) ~len:head_size in
          let a = att.(t) in
          for i = 0 to head_size - 1 do
            xb.(i) <- xb.(i) +. (a *. v.(i));
          done;
        done;
      done;
      matmul state.RunState.xb2 state.RunState.xb weights.TransformerWeights.wo.(l) dim dim;
      for i = 0 to dim - 1 do
        state.RunState.x.(i) <- state.RunState.x.(i) +. state.RunState.xb2.(i);
      done;
      rmsnorm state.RunState.xb state.RunState.x weights.TransformerWeights.rms_ffn_weight.(l);
      matmul state.RunState.hb state.RunState.xb weights.TransformerWeights.w1.(l) dim hidden_dim;
      matmul state.RunState.hb2 state.RunState.xb weights.TransformerWeights.w3.(l) dim hidden_dim;
      for i = 0 to hidden_dim - 1 do
        let val_ = state.RunState.hb.(i) in
        val_ <- val_ *. (1.0 /. (1.0 +. exp (~-.val_)));
        val_ <- val_ *. state.RunState.hb2.(i);
        state.RunState.hb.(i) <- val_;
      done;
      matmul state.RunState.xb state.RunState.hb weights.TransformerWeights.w2.(l) hidden_dim dim;
      for i = 0 to dim - 1 do
        state.RunState.x.(i) <- state.RunState.x.(i) +. state.RunState.xb.(i);
      done;
    done;
    rmsnorm state.RunState.x state.RunState.x weights.TransformerWeights.rms_final_weight;
    matmul state.RunState.logits state.RunState.x (Option.value_exn weights.TransformerWeights.wcls) config.Config.dim config.Config.vocab_size;
    state.RunState.logits

  let create checkpoint_path =
    let fd = Unix.openfile checkpoint_path [Unix.O_RDONLY] 0o400 in
    let file_size = Unix.((fstat fd).st_size) in
    let data = Array.create ~len:file_size 0.0 in
    let _bytes_read = Unix.read fd data 0 file_size in
    Unix.close fd;
    let config_size = sizeof_int * 7 in
    let config_data = Array.sub data ~pos:0 ~len:config_size in
    let config = {
      Config.dim = int_of_bytes (Array.sub config_data ~pos:0 ~len:sizeof_int);
      hidden_dim = int_of_bytes (Array.sub config_data ~pos:sizeof_int ~len:sizeof_int);
      n_layers = int_of_bytes (Array.sub config_data ~pos:(2 * sizeof_int) ~len:sizeof_int);
      n_heads = int_of_bytes (Array.sub config_data ~pos:(3 * sizeof_int) ~len:sizeof_int);
      n_kv_heads = int_of_bytes (Array.sub config_data ~pos:(4 * sizeof_int) ~len:sizeof_int);
      vocab_size = abs (int_of_bytes (Array.sub config_data ~pos:(5 * sizeof_int) ~len:sizeof_int));
      seq_len = int_of_bytes (Array.sub config_data ~pos:(6 * sizeof_int) ~len:sizeof_int);
    } in
    let shared_weights = config.Config.vocab_size > 0 in
    let weights_ptr = Array.sub data ~pos:config_size ~len:(Array.length data - config_size) in
    let weights = {
      TransformerWeights.token_embedding_table = Array.sub weights_ptr ~pos:0 ~len:(config.Config.vocab_size * config.Config.dim);
      rms_att_weight = Array.init config.Config.n_layers ~f:(fun l -> Array.sub weights_ptr ~pos:(config.Config.vocab_size * config.Config.dim + l * config.Config.dim) ~len:config.Config.dim);
      wq = Array.init config.Config.n_layers ~f:(fun l -> Array.sub weights_ptr ~pos:(config.Config.vocab_size * config.Config.dim + config.Config.n_layers * config.Config.dim + l * config.Config.dim * (config.Config.n_heads * (config.Config.dim / config.Config.n_heads))) ~len:(config.Config.dim * (config.Config.n_heads * (config.Config.dim / config.Config.n_heads))));
      wk = Array.init config.Config.n_layers ~f:(fun l -> Array.sub weights_ptr ~pos:(config.Config.vocab_size * config.Config.dim + config.Config.n_layers * config.Config.dim + config.Config.n_layers * config.Config.dim * (config.Config.n_heads * (config.Config.dim / config.Config.n_heads)) + l * config.Config.dim * (config.Config.n_kv_heads * (config.Config.dim / config.Config.n_heads))) ~len:(config.Config.dim * (config.Config.n_kv_heads * (config.Config.dim / config.Config.n_heads))));
      wv = Array.init config.Config.n_layers ~f:(fun l -> Array.sub weights_ptr ~pos:(config.Config.vocab_size * config.Config.dim + config.Config.n_layers * config.Config.dim + config.Config.n_layers * config.Config.dim * (config.Config.n_heads * (config.Config.dim / config.Config.n_heads)) + config.Config.n_layers * config.Config.dim * (config.Config.n_kv_heads * (config.Config.dim / config.Config.n_heads)) + l * config.Config.dim * (config.Config.n_kv_heads * (config.Config.dim / config.Config.n_heads))) ~len:(config.Config.dim * (config.Config.n_kv_heads * (config.Config.dim / config.Config.n_heads))));
      wo = Array.init config.Config.n_layers ~f:(fun l -> Array.sub weights_ptr ~pos:(config.Config.vocab_size * config.Config.dim + config.Config.n_layers * config.Config.dim + config.Config.n_layers * config.Config.dim * (config.Config.n_heads * (config.Config.dim / config.Config.n_heads)) + config.Config.n_layers * config.Config.dim * (config.Config.n_kv_heads * (config.Config.dim / config.Config.n_heads)) + config.Config.n_layers * config.Config.dim * (config.Config.n_kv_heads * (config.Config.dim / config.Config.n_heads)) + l * (config.Config.n_heads * (config.Config.dim / config.Config.n_heads)) * config.Config.dim) ~len:((config.Config.n_heads * (config.Config.dim / config.Config.n_heads)) * config.Config.dim));
      rms_ffn_weight = Array.init config.Config.n_layers ~f:(fun l -> Array.sub weights_ptr ~pos:(config.Config.vocab_size * config.Config.dim + config.Config.n_layers * config.Config.dim + config.Config.n_layers * config.Config.dim * (config.Config.n_heads * (config.Config.dim / config.Config.n_heads)) + config.Config.n_layers * config.Config.dim * (config.Config.n_kv_heads * (config.Config.dim / config.Config.n_heads)) + config.Config.n_layers * config.Config.dim * (config.Config.n_kv_heads * (config.Config.dim / config.Config.n_heads)) + config.Config.n_layers * (config.Config.n_heads * (config.Config.dim / config.Config.n_heads)) * config.Config.dim + l * config.Config.dim) ~len:config.Config.dim);
      w1 = Array.init config.Config.n_layers ~f:(fun l -> Array.sub weights_ptr ~pos:(config.Config.vocab_size * config.Config.dim + config.Config.n_layers * config.Config.dim + config.Config.n_layers * config.Config.dim * (config.Config.n_heads * (config.Config.dim / config.Config.n_heads)) + config.Config.n_layers * config.Config.dim * (config.Config.n_kv_heads * (config.Config.dim / config.Config.n_heads)) + config.Config.n_layers * config.Config.dim * (config.Config.n_kv_heads * (config.Config.dim / config.Config.n_heads)) + config.Config.n_layers * (config.Config.n_heads * (config.Config.dim / config.Config.n_heads)) * config.Config.dim + config.Config.n_layers * config.Config.dim + l * config.Config.dim * config.Config.hidden_dim) ~len:(config.Config.dim * config.Config.hidden_dim));
      w2 = Array.init config.Config.n_layers ~f:(fun l -> Array.sub weights_ptr ~pos:(config.Config.vocab_size * config.Config.dim + config.Config.n_layers * config.Config.dim + config.Config.n_layers * config.Config.dim * (config.Config.n_heads * (config.Config.dim / config.Config.n_heads)) + config.Config.n_layers * config.Config.dim * (config.Config.n_kv_heads * (config.Config.dim / config.Config.n_heads)) + config.Config.n_layers * config.Config.dim * (config.Config.n_kv_heads * (config.Config.dim / config.Config.n_heads)) + config.Config.n_layers * (config.Config.n_heads * (config.Config.dim / config.Config.n_heads)) * config.Config.dim + config.Config.n_layers * config.Config.dim + config.Config.n_layers * config.Config.dim * config.Config.hidden_dim + l * config.Config.hidden_dim * config.Config.dim) ~len:(config.Config.hidden_dim * config.Config.dim));
      w3 = Array.init config.Config.n_layers ~f:(fun l -> Array.sub weights_ptr ~pos:(config.Config.vocab_size * config.Config.dim + config.Config.n_layers * config.Config.dim + config.Config.n_layers * config.Config.dim * (config.Config.n_heads * (config.Config.dim / config.Config.n_heads)) + config.Config.n_layers * config.Config.dim * (config.Config.n_kv_heads * (config.Config.dim / config.Config.n_heads)) + config.Config.n_layers * config.Config.dim * (config.Config.n_kv_heads * (config.Config.dim / config.Config.n_heads)) + config.Config.n_layers * (config.Config.n_heads * (config.Config.dim / config.Config.n_heads)) * config.Config.dim + config.Config.n_layers * config.Config.dim + config.Config.n_layers * config.Config.dim * config.Config.hidden_dim + config.Config.n_layers * config.Config.hidden_dim * config.Config.dim + l * config.Config.dim * config.Config.hidden_dim) ~len:(config.Config.dim * config.Config.hidden_dim));
      rms_final_weight = Array.sub weights_ptr ~pos:(config.Config.vocab_size * config.Config.dim + config.Config.n_layers * config.Config.dim + config.Config.n_layers * config.Config.dim * (config.Config.n_heads * (config.Config.dim / config.Config.n_heads)) + config.Config.n_layers * config.Config.dim * (config.Config.n_kv_heads * (config.Config.dim / config.Config.n_heads)) + config.Config.n_layers * config.Config.dim * (config.Config.n_kv_heads * (config.Config.dim / config.Config.n_heads)) + config.Config.n_layers * (config.Config.n_heads * (config.Config.dim / config.Config.n_heads)) * config.Config.dim + config.Config.n_layers * config.Config.dim + config.Config.n_layers * config.Config.dim * config.Config.hidden_dim + config.Config.n_layers * config.Config.hidden_dim * config.Config.dim + config.Config.n_layers * config.Config.dim * config.Config.hidden_dim) ~len:config.Config.dim;
      wcls = if shared_weights then Some weights.TransformerWeights.token_embedding_table else None;
      } in
      let state = RunState.create config in
      { config; weights; state; fd; data; file_size }
      end
      
      (* Tokenizer *)
      module Tokenizer = struct
      type t = {
      vocab : string array;
      vocab_scores : float array;
      sorted_vocab : (string * int) array;
      vocab_size : int;
      max_token_length : int;
      byte_pieces : char array;
      } [@@deriving sexp]
      
      let compare_tokens (a : string * int) (b : string * int) =
      String.compare (fst a) (fst b)
      
      let create tokenizer_path vocab_size =
      let vocab = Array.create ~len:vocab_size "" in
      let vocab_scores = Array.create ~len:vocab_size 0.0 in
      let byte_pieces = Array.create ~len:512 '\000' in
      for i = 0 to 255 do
      byte_pieces.(i * 2) <- Char.of_int_exn i;
      byte_pieces.(i * 2 + 1) <- '\000';
      done;
      let file = In_channel.create tokenizer_path in
      let max_token_length = In_channel.input_binary_int file in
      for i = 0 to vocab_size - 1 do
      vocab_scores.(i) <- In_channel.input_binary_float file;
      let len = In_channel.input_binary_int file in
      vocab.(i) <- In_channel.really_input_string file len;
      done;
      In_channel.close file;
      let sorted_vocab = Array.mapi vocab ~f:(fun i s -> (s, i)) in
      Array.sort sorted_vocab ~cmp:compare_tokens;
      { vocab; vocab_scores; sorted_vocab; vocab_size; max_token_length; byte_pieces }
      
      let encode t text bos eos tokens n_tokens =
      let str_buffer = Bytes.create (t.max_token_length * 2 + 1 + 2) in
      let str_len = ref 0 in
      n_tokens := 0;
      if bos then (
      tokens.(0) <- 1;
      n_tokens := !n_tokens + 1;
      );
      if String.length text > 0 then (
      let dummy_prefix = fst (Array.find_exn t.sorted_vocab ~f:(fun (s, _) -> s = " ")) in
      tokens.(!n_tokens) <- snd dummy_prefix;
      n_tokens := !n_tokens + 1;
      );
      let rec encode_utf8 c =
      if Char.code c < 0x80 then (
      Bytes.set str_buffer !str_len c;
      str_len := !str_len + 1;
      )
      else if Char.code c < 0x800 then (
      Bytes.set str_buffer !str_len (Char.of_int_exn (0xC0 lor (Char.code c lsr 6)));
      str_len := !str_len + 1;
      Bytes.set str_buffer !str_len (Char.of_int_exn (0x80 lor (Char.code c land 0x3F)));
      str_len := !str_len + 1;
      )
      else if Char.code c < 0x10000 then (
      Bytes.set str_buffer !str_len (Char.of_int_exn (0xE0 lor (Char.code c lsr 12)));
      str_len := !str_len + 1;
      Bytes.set str_buffer !str_len (Char.of_int_exn (0x80 lor ((Char.code c lsr 6) land 0x3F)));
      str_len := !str_len + 1;
      Bytes.set str_buffer !str_len (Char.of_int_exn (0x80 lor (Char.code c land 0x3F)));
      str_len := !str_len + 1;
      )
      else (
      Bytes.set str_buffer !str_len (Char.of_int_exn (0xF0 lor (Char.code c lsr 18)));
      str_len := !str_len + 1;
      Bytes.set str_buffer !str_len (Char.of_int_exn (0x80 lor ((Char.code c lsr 12) land 0x3F)));
      str_len := !str_len + 1;
      Bytes.set str_buffer !str_len (Char.of_int_exn (0x80 lor ((Char.code c lsr 6) land 0x3F)));
      str_len := !str_len + 1;
      Bytes.set str_buffer !str_len (Char.of_int_exn (0x80 lor (Char.code c land 0x3F)));
      str_len := !str_len + 1;
      )
      in
      String.iter text ~f:encode_utf8;
      Bytes.set str_buffer !str_len '\000';
      let rec encode_tokens () =
      let str = Bytes.to_string (Bytes.sub str_buffer ~pos:0 ~len:!str_len) in
      match Array.binary_search t.sorted_vocab ~compare:compare_tokens (str, 0) with
      | `Found index ->
      tokens.(!n_tokens) <- snd t.sorted_vocab.(index);
      n_tokens := !n_tokens + 1;
      str_len := 0;
      | `Not_found _ ->
      for i = 0 to !str_len - 1 do
      tokens.(!n_tokens) <- Char.code (Bytes.get str_buffer i) + 3;
      n_tokens := !n_tokens + 1;
      done;
      str_len := 0;
      in
      while !str_len > 0 do
      encode_tokens ();
      done;
      while true do
      let best_score = ref (-1e10) in
      let best_id = ref (-1) in
      let best_idx = ref (-1) in
      for i = 0 to !n_tokens - 2 do
      let merged = sprintf "%s%s" t.vocab.(tokens.(i)) t.vocab.(tokens.(i + 1)) in
      match Array.binary_search t.sorted_vocab ~compare:compare_tokens (merged, 0) with
      | `Found index ->
      if t.vocab_scores.(snd t.sorted_vocab.(index)) > !best_score then (
      best_score := t.vocab_scores.(snd t.sorted_vocab.(index));
      best_id := snd t.sorted_vocab.(index);
      best_idx := i;
      )
      | `Not_found _ -> ()
      done;
      if !best_idx = -1 then break;
      tokens.(!best_idx) <- !best_id;
      for i = !best_idx + 1 to !n_tokens - 2 do
      tokens.(i) <- tokens.(i + 1);
      done;
      n_tokens := !n_tokens - 1;
      done;
      if eos then (
      tokens.(!n_tokens) <- 2;
      n_tokens := !n_tokens + 1;
      );
      ()
      
      let decode t prev_token token =
      let piece =
      if token = 1 && t.vocab.(prev_token).[0] = ' ' then
      String.sub t.vocab.(token) ~pos:1 ~len:(String.length t.vocab.(token) - 1)
      else
      t.vocab.(token)
      in
      match Scanf.sscanf piece "<0x%02x>" (fun b -> b) with
      | Some byte_val -> String.make 1 (Char.of_int_exn byte_val)
      | None -> piece
      end
      
      (* Sampler *)
      module Sampler = struct
      type t = {
      vocab_size : int;
      probindex : (float * int) array;
      temperature : float;
      topp : float;
      rng_state : Int64.t;
      } [@@deriving sexp]
      
      let sample_argmax probabilities n =
      let max_i = ref 0 in
      let max_p = ref probabilities.(0) in
      for i = 1 to n - 1 do
      if probabilities.(i) > !max_p then (
      max_i := i;
      max_p := probabilities.(i);
      )
      done;
      !max_i
      
      let sample_mult probabilities n coin =
      let cdf = ref 0.0 in
      for i = 0 to n - 1 do
      cdf := !cdf +. probabilities.(i);
      if coin < !cdf then (
      return i
      )
      done;
      n - 1
      
      let compare (a : float * int) (b : float * int) =
      if fst a > fst b then -1
      else if fst a < fst b then 1
      else 0
      
      let sample_topp probabilities n topp probindex coin =
      let n0 = ref 0 in
      let cutoff = (1.0 -. topp) /. Float.of_int (n - 1) in
      for i = 0 to n - 1 do
      if probabilities.(i) >= cutoff then (
      probindex.(!n0) <- (probabilities.(i), i);
      n0 := !n0 + 1;
      )
      done;
      Array.sort probindex ~cmp:compare;
      let cumulative_prob = ref 0.0 in
      let last_idx = ref (!n0 - 1) in
      for i = 0 to !n0 - 1 do
      cumulative_prob := !cumulative_prob +. fst probindex.(i);
      if !cumulative_prob > topp then (
      last_idx := i;
      break;
      )
      done;
      let r = coin *. !cumulative_prob in
      let cdf = ref 0.0 in
      for i = 0 to !last_idx do
      cdf := !cdf +. fst probindex.(i);
      if r < !cdf then (
      return snd probindex.(i)
      )
      done;
      snd probindex.(!last_idx)
      
      let create vocab_size temperature topp rng_seed =
        {
          vocab_size;
          probindex = Array.create ~len:vocab_size (0.0, 0);
          temperature;
          topp;
          rng_state = Int64.of_int rng_seed;
        }
      
      let sample t logits =
        let probabilities = Array.copy logits in
        if t.temperature = 0.0 then
          sample_argmax probabilities t.vocab_size
        else (
          Array.iteri probabilities ~f:(fun i p -> probabilities.(i) <- p /. t.temperature);
          Transformer.softmax probabilities;
          let coin = Random.State.float (Random.State.make [| t.rng_state |]) 1.0 in
          if t.topp <= 0.0 || t.topp >= 1.0 then
            sample_mult probabilities t.vocab_size coin
          else
            sample_topp probabilities t.vocab_size t.topp t.probindex coin
        )
      end
      
      (* Main program *)
      let () =
        let checkpoint_path = Sys.argv.(1) in
        let temperature = 1.0 in
        let topp = 0.9 in
        let steps = 256 in
        let prompt = None in
        let rng_seed = Time_now.nanoseconds_since_unix_epoch () |> Int64.to_int in
        let mode = "generate" in
        let system_prompt = None in
      
        (* Parameter validation/overrides *)
        (* TODO... *)
      
        (* Build the Transformer *)
        let transformer = Transformer.create checkpoint_path in
      
        (* Build the Tokenizer *)
        let tokenizer = Tokenizer.create "tokenizer.bin" transformer.config.Config.vocab_size in
      
        (* Build the Sampler *)
        let sampler = Sampler.create transformer.config.Config.vocab_size temperature topp rng_seed in
      
        (* Run the model *)
        match mode with
        | "generate" ->
          let prompt_str = Option.value prompt ~default:"" in
          let prompt_tokens = Array.create ~len:(String.length prompt_str + 3) 0 in
          let n_prompt_tokens = ref 0 in
          Tokenizer.encode tokenizer prompt_str true false prompt_tokens n_prompt_tokens;
          let token = ref prompt_tokens.(0) in
          let pos = ref 0 in
          while !pos < steps do
            let logits = Transformer.forward transformer !token !pos in
            let next =
              if !pos < !n_prompt_tokens - 1 then
                prompt_tokens.(!pos + 1)
              else
                Sampler.sample sampler logits
            in
            incr pos;
            if next = 1 then break;
            let piece = Tokenizer.decode tokenizer !token next in
            printf "%s" piece;
            token := next
          done;
          printf "\n"
        | _ ->
          failwith "Unknown mode"