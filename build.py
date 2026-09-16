#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build qwen38-27b-tensor-atlas/index.html.

Method: the nisten "Tensor Atlas v4" engine (src.html = nisten/LLMViz-DeepSeek-V4.1-Flash,
MIT) is kept whole for the shell — topbar, view tabs, WebGL2 instanced-cube renderer,
transport/timeline, inspector, storage ledger, safetensors audit, help/licence modals, CSS.
Only the model half is replaced: the data+logic region (`const CFG` … `const VOLUME_UNIT`)
becomes Qwen3.8-27B, plus the copy strings and the precision pair.

Data: hf-bf16.json / hf-fp8.json — every tensor's byte length read from the safetensors
headers of Qwen/Qwen3.8-27B and Qwen/Qwen3.8-27B-FP8 over HTTP Range (no weights downloaded).
The FP8 repo's 128x128 block scales are folded into the tensor they scale, so every byte
figure on the page is a real on-disk size, asserted against the repositories' own totals.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
SRC = DIR / "src.html"
OUT = DIR / "index.html"
BF_JSON = DIR / "hf-bf16.json"
F8_JSON = DIR / "hf-fp8.json"

BF_REPO = "Qwen/Qwen3.8-27B"
F8_REPO = "Qwen/Qwen3.8-27B-FP8"

# ─────────────────────────────── data plumbing ───────────────────────────────
def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def collect(d: dict) -> dict:
    out = {}
    for n, t in d["tensors"].items():
        lg = re.sub(r"\.(weight_scale_inv|scale_inv|input_scale)$", ".weight", n)
        rec = out.setdefault(lg, {"bytes": 0, "dims": t["shape"]})
        rec["bytes"] += t["bytes"]
    return out


def templates():
    bf, f8 = collect(load(BF_JSON)), collect(load(F8_JSON))
    missing = [n for n in bf if n not in f8]
    if missing:
        raise SystemExit(f"tensors missing from the FP8 repo: {missing[:3]}")
    both = {n: (bf[n]["bytes"], f8[n]["bytes"], bf[n]["dims"]) for n in bf}

    def lin_of(i: int) -> dict:
        pre = f"model.language_model.layers.{i}."
        return {n[len(pre):]: v for n, v in both.items() if n.startswith(pre)}

    t_lin, t_full = lin_of(0), lin_of(3)
    for i in range(64):
        got = lin_of(i)
        want = t_full if i % 4 == 3 else t_lin
        if set(got) != set(want) or any(got[k][:2] != want[k][:2] for k in got):
            raise SystemExit(f"layer {i} differs from its archetype")

    vis_groups: dict[str, dict] = {}
    for n, (b16, b8, dims) in both.items():
        if not n.startswith("model.visual."):
            continue
        key = re.sub(r"blocks\.\d+\.", "blocks.*.", n.replace("model.visual.", ""))
        g = vis_groups.setdefault(key, {"b16": 0, "b8": 0, "dims": list(dims), "count": 0})
        g["b16"] += b16
        g["b8"] += b8
        g["count"] += 1

    io = {
        "embed": [("model.language_model.embed_tokens.weight", *both["model.language_model.embed_tokens.weight"])],
        "head": [("model.language_model.norm.weight", *both["model.language_model.norm.weight"]),
                 ("lm_head.weight", *both["lm_head.weight"])],
        "mtp": [],
    }
    for n in sorted(both):
        if n.startswith("mtp.layers.0."):
            io["mtp"].append((n.replace("mtp.layers.0.", ""), *both[n]))
    for n in ("mtp.fc.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight"):
        io["mtp"].append((n, *both[n]))
    order = {"mtp.fc.weight": 0, "mtp.pre_fc_norm_embedding.weight": 1, "mtp.pre_fc_norm_hidden.weight": 2}
    io["mtp"].sort(key=lambda it: (order.get(it[0], 3), it[0]))
    vis = sorted(((k, v) for k, v in vis_groups.items()), key=lambda kv: -kv[1]["b16"])
    return {"lin": t_lin, "full": t_full, "vis": vis, "io": io}


CAT_BY_PATTERN = [
    (r"embed_tokens", "embed"),
    (r"^lm_head", "head"),
    (r"norm", "norm"),
    (r"linear_attn", "dnet"),
    (r"self_attn", "attn"),
    (r"mlp\.", "ffn"),
    (r"^mtp", "mtp"),
]


def cat_of(name: str, kind: str) -> str:
    if kind == "vis":
        return "vmerge" if re.search(r"merger|patch_embed|pos_embed", name) else "vision"
    for pat, c in CAT_BY_PATTERN:
        if re.search(pat, name):
            return c
    return "norm"


NOTE = {
    "input_layernorm.weight": "pre-mixer RMSNorm",
    "post_attention_layernorm.weight": "pre-FFN RMSNorm",
    "linear_attn.conv1d.weight": "depthwise · kernel 4",
    "linear_attn.in_proj_qkv.weight": "16 QK + 48 V heads, fused",
    "linear_attn.in_proj_z.weight": "swish output gate",
    "linear_attn.in_proj_b.weight": "β write gate",
    "linear_attn.in_proj_a.weight": "α decay projection",
    "linear_attn.A_log": "learned base decay",
    "linear_attn.dt_bias": "Δt bias",
    "linear_attn.norm.weight": "per-head state norm",
    "linear_attn.out_proj.weight": "back to the 5 120-dim stream",
    "self_attn.q_proj.weight": "Q + fused output gate → 2× wide",
    "self_attn.q_norm.weight": "per-head QK-norm",
    "self_attn.k_proj.weight": "4 KV heads · GQA",
    "self_attn.k_norm.weight": "per-head QK-norm",
    "self_attn.v_proj.weight": "4 KV heads · GQA",
    "self_attn.o_proj.weight": "24 head outputs → hidden",
    "mlp.gate_proj.weight": "SwiGLU gate 5 120 → 17 408",
    "mlp.up_proj.weight": "SwiGLU up 5 120 → 17 408",
    "mlp.down_proj.weight": "SwiGLU down 17 408 → 5 120",
    "mtp.fc.weight": "embed ⊕ hidden → hidden",
    "embed_tokens.weight": "lookup 248 320 × 5 120",
    "lm_head.weight": "unembedding 5 120 → 248 320",
}


def js_weight_array(var: str, items, kind: str, comment: str) -> str:
    lines = [f"const {var}=[", f"  /* {comment} */"]
    for it in items:
        name, b16, b8, dims = it[0], it[1], it[2], it[3]
        count = it[4] if len(it) > 4 else 1
        note = NOTE.get(name, "")
        extra = f",{count}" if count > 1 else ""
        lines.append(
            f"  W({json.dumps(name)},{json.dumps(dims)},{json.dumps(cat_of(name, kind))},"
            f"{json.dumps(note)},{{bf16:{b16},fp8:{b8}}}{extra}),"
        )
    lines.append("];")
    return "\n".join(lines)


# ─────────────────────────── the model half (JS) ───────────────────────────
DATA_JS = """// ---------- qwen38-27b data.js ----------
/* Qwen3.8-27B Tensor Atlas. Model facts come from the published config.json of
   Qwen/Qwen3.8-27B and its official FP8 build; every byte total is audited against the
   safetensors headers of those two repositories (read over HTTP Range, summed per tensor,
   block scales folded into the tensor they scale). Not schema-derived. */
const CFG = {
 text_config:{
  hidden_size:5120,num_hidden_layers:64,num_attention_heads:24,num_key_value_heads:4,
  head_dim:256,partial_rotary_factor:0.25,rope_theta:10000000,rms_norm_eps:1e-6,
  intermediate_size:17408,vocab_size:248320,max_position_embeddings:262144,
  full_attention_interval:4,attn_output_gate:true,output_gate_type:'swish',
  linear_conv_kernel_dim:4,linear_key_head_dim:128,linear_num_key_heads:16,
  linear_num_value_heads:48,linear_value_head_dim:128,mamba_ssm_dtype:'float32',
  mtp_num_hidden_layers:1,tie_word_embeddings:false,
  layer_types:Array.from({length:64},(_,i)=>i%4===3?'full_attention':'linear_attention'),
  /* [LIN,LIN,LIN,FULL] x 16 — the 3:1 rhythm, and the only thing DeepSeek-style
     kv_source/index_source plumbing is replaced by here (see modeFor below) */
  ctx_extended:1000000
 },
 vision_config:{depth:27,hidden_size:1152,num_heads:16,head_dim:72,intermediate_size:4304,
  patch_size:16,spatial_merge_size:2,temporal_patch_size:2,num_position_embeddings:2304,
  out_hidden_size:5120,hidden_act:'gelu_pytorch_tanh',deepstack_visual_indexes:[]},
 quantization_config:{quant_method:'fp8',fmt:'e4m3',activation_scheme:'dynamic',
  weight_block_size:[128,128],
  ignored:'embed_tokens · lm_head · the whole vision tower · mtp.fc · linear_attn.{A_log,dt_bias,conv1d,in_proj_a,in_proj_b} · every norm and bias'}
};
const CT=CFG.text_config;
const VC=CFG.vision_config;
const COL={blue:'#638bff',enc:'#5ca7ff',dec:'#55d7c1',engram:'#b99bff',expert:'#76b9ff',shared:'#d6e99c',attn:'#55ddd0',router:'#f4b765',mhc:'#e3a8dc',vision:'#80ceea',head:'#c4d0ff',full:'#efbc71',reindex:'#b798ff',reuse:'#709bbd',swa:'#758090',norm:'#a7b5cc',muted:'#8390a6'};
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const lerp=(a,b,t)=>a+(b-a)*t;
const smooth=x=>x*x*(3-2*x);
const num=x=>Math.round(x).toLocaleString('en-US');
function fmtP(p){return p>=1e12?(p/1e12).toFixed(3)+'T':p>=1e9?(p/1e9).toFixed(2)+'B':p>=1e6?(p/1e6).toFixed(2)+'M':p>=1e3?(p/1e3).toFixed(1)+'K':num(p);}
function bytes(n,binary=false){let b=binary?1024:1000,u=binary?['B','KiB','MiB','GiB','TiB']:['B','KB','MB','GB','TB'],k=0;while(n>=b&&k<4){n/=b;k++;}return (k===0?num(n):n.toFixed(n>=100?1:2))+' '+u[k];}
const SOURCE_BASE='https://huggingface.co/Qwen/Qwen3.8-27B';
const SOURCES=[
 ['Model card',SOURCE_BASE,'27B dense vision-language model, 64 layers, 262,144-token context, Apache-2.0. Reported benchmarks and the 1M-token extension live here.'],
 ['Released config',SOURCE_BASE+'/blob/main/config.json','The text and vision configs are embedded field-for-field: hybrid layer_types, Gated DeltaNet head counts, partial rotary 0.25 with mrope sections, the 27-block vision tower and the single MTP layer.'],
 ['Official FP8 build','https://huggingface.co/Qwen/Qwen3.8-27B-FP8','The quantized checkpoint this page toggles to: e4m3 weights, one F32 scale per 128x128 block, dynamic activations, and a modules_to_not_convert list that decides exactly what stays BF16.'],
 ['Safetensors headers',SOURCE_BASE+'/tree/main','Every byte on this page was read from these shards over HTTP Range (18 BF16 shards, 66 FP8 shards) and summed per tensor. No weights were downloaded.'],
 ['Vision tower',SOURCE_BASE+'/blob/main/preprocessor_config.json','Patch-size 16, temporal patch 2, spatial merge 2, 2,304 learned positions and a 4,608 -> 5,120 merger back into the language model.'],
 ['Licence',SOURCE_BASE+'/blob/main/LICENSE','Apache-2.0 for the model and its weights. The page engine is MIT, © 2026 netsin. No weight data is bundled here — metadata only.']
];
const MODE_INFO={bf16:{label:'BF16 checkpoint',short:'BF16',color:COL.enc,note:'Qwen/Qwen3.8-27B · 18 shards · 2.00 bytes per parameter'},
                 fp8:{label:'Official FP8',short:'FP8',color:COL.dec,note:'Qwen/Qwen3.8-27B-FP8 · e4m3 · 128×128 block scales · vision + embeddings stay BF16'}};
const MODE_KEYS=['bf16','fp8'];
/* this model is dense: nothing is skipped, so "mode" here is only about memory shape —
   a full-attention layer owns a KV cache, a Gated DeltaNet layer owns a constant state */
function modeFor(i){return i%4===3?'full':'reuse';}
function ownerFor(i){return i%4===3?i:null;}
function indexOwnerFor(i){return null;}
function W(name,shape,cat,note='',ex=null,count=1){
 let p=shape.reduce((a,b)=>a*b,1)*count;
 return {name,shape,format:(ex&&ex.fp8<ex.bf16*0.75)?'fp8':'bf16',cat,count,p,note,ex};
}
function wBytes(w,mode='bf16'){
 if(w.ex)return w.ex[mode];
 return 2*w.p;   /* every weight in this checkpoint carries its own measured bytes */
}
function wFormat(w,mode){return mode==='fp8'?(w.format==='fp8'?'FP8 / 128×128':'BF16'):'BF16';}
const sumP=ws=>ws.reduce((a,w)=>a+w.p,0);
const sumB=(ws,m)=>ws.reduce((a,w)=>a+wBytes(w,m),0);
@@TABLES@@
function layerWeights(i){return i%4===3?FULL_W:LIN_W;}
const LAYERS=Array.from({length:64},(_,i)=>({id:'L'+i,index:i,label:'Layer '+String(i).padStart(2,'0'),
 part:i%4===3?'attention':'linear',mode:modeFor(i),owner:ownerFor(i),indexOwner:indexOwnerFor(i),
 ratio:1,ws:layerWeights(i)}));
const ENGRAM=[];                       /* no Engram memory in this model */
const DRAFT=[];                        /* no DSpark stages in this model */
const EMBED={id:'embed',label:'Token embedding',ws:[W('model.language_model.embed_tokens.weight',[248320,5120],'embed','2.54 GB, untied from the head. BF16 in both builds — every token touches it.')]};
const HEAD={id:'head',label:'LM head',ws:[W('model.language_model.norm.weight',[5120],'norm','final RMSNorm'),W('lm_head.weight',[248320,5120],'head','Untied 5,120 -> 248,320 logits. Kept BF16 by the FP8 build.')]};
const MTP={id:'mtp',label:'MTP head',index:64,ws:MTP_W};
const VISION={id:'vision',label:'Vision tower',ws:VISION_W};
const MODULES=[EMBED,...LAYERS,HEAD,MTP,VISION];
const ALL_W=MODULES.flatMap(m=>m.ws);
const TOTALS=Object.fromEntries(MODE_KEYS.map(m=>[m,sumB(ALL_W,m)]));
const TOTAL_P=sumP(ALL_W);
const MOE_EXPERT_P=0;
const MOE_TOTAL_P=0;
const NV_DELTA=TOTALS.fp8-TOTALS.bf16;
const CATEGORIES=[
 ['ffn','Dense SwiGLU FFN',COL.expert,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='ffn'))],
 ['dnet','Gated DeltaNet mixers',COL.dec,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='dnet'))],
 ['attn','Gated attention',COL.attn,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='attn'))],
 ['vision','Vision tower',COL.vision,VISION.ws],
 ['mtp','MTP head',COL.engram,MTP.ws],
 ['vocab','Embedding + head',COL.enc,[...EMBED.ws,...HEAD.ws]],
 ['norm','Norms',COL.norm,ALL_W.filter(w=>w.cat==='norm')]
];
const EXP={
 overview:{title:'Sixty-four dense blocks, two kinds of memory.',body:'Qwen3.8-27B is a 27.8-billion-parameter vision-language model with no mixture of experts anywhere: every block ends in one wide SwiGLU feed-forward that runs for every token. What varies is the mixer — three of every four layers run a Gated DeltaNet recurrence whose state is a constant 3.1 MB, and every fourth layer runs gated softmax attention that keeps a KV cache. That 3:1 rhythm is the whole memory story on this page; the 34.2 GB feed-forward is the more brutal one.'},
 dnet:{title:'A recurrence with no cache at all.',body:'The linear mixer fuses 16 query/key heads and 48 value heads at 128 dimensions each, smooths them with a kernel-4 causal convolution, then read-modify-writes one 128x128 state per value head with a delta rule — steered by a learned decay (A_log, dt_bias, the alpha projection) and a data-dependent write gate beta, and filtered on the way out by a swish gate. Per token the cost is O(1): 48 x 128 x 128 F32 = 3.1 MB per layer, unchanged whether the context is ten tokens or 262,144. This is 48 of the 64 layers.'},
 full:{title:'One layer in four pays rent on the whole context.',body:'Gated attention over the full window: 24 query heads share 4 KV heads at head_dim 256, with per-head RMSNorm on Q and K. Only a quarter of each head carries position (partial rotary 0.25, mrope sections 11/11/10, theta 10 million). q_proj is twice as wide as it looks like it should be — 5,120 x 12,288 — because the per-head swish output gate is fused into it. A token costs 4 KB of KV per layer here, and these 16 layers are the only ones that grow with context: 17.2 GB at 262K, 65.5 GB at the 1M extension.'},
 vision:{title:'A native VLM encoder, not a bolt-on.',body:'Images arrive as 16x16 patches across two frames, take a learned 2,304-position table, and run through 27 transformer blocks of hidden 1,152 — 16 heads at head_dim 72, a fused qkv projection and a 4,304-dim GELU-tanh MLP. The merger folds 2x2 spatial neighbours (4,608 = 4 x 1,152) and projects back to 5,120, the language model hidden size, so after the merger image tokens are ordinary tokens. 921.5 MB of the checkpoint, and BF16 in both published builds.'},
 mtp:{title:'A speculative decoder bolted to the trunk.',body:'One extra full-attention block with its own dense feed-forward, plus a 10,240 -> 5,120 fusion projection that concatenates the pre-normed embedding of the just-predicted token with the trunk final hidden state. Trained with multiple MTP steps, it drafts the token after next for self-speculative decoding with no separate drafter model. 849.4 MB, sharing the trunk embedding table, and quantized in the FP8 build while its fusion fc and norms stay BF16.'},
 bf16:{title:'The reference checkpoint.',body:'55.56 GB across 18 shards: every one of the 1,199 tensors at two bytes per parameter. The two vocabulary tables are 5.09 GB of that, the dense feed-forward is 34.2 GB, the Gated DeltaNet mixers add 9.6 GB, attention 3.9 GB, the vision tower 0.92 GB and the MTP head 0.85 GB.'},
 fp8:{title:'The official FP8 build, byte for byte.',body:'30.87 GB across 66 shards and 1,606 tensors — 1.80x smaller, not 4x, because the repository does not quantize everything. Read its own modules_to_not_convert list and you get this page precision toggle: both embedding tables, lm_head, the entire 27-block vision tower, mtp.fc, the DeltaNet gates (A_log, dt_bias, conv1d, the alpha and beta projections) and every RMSNorm and bias stay BF16; the big matmuls become e4m3 with one F32 scale per 128x128 block (1.00024 bytes per parameter) and dynamic activations.'},
 storage:{title:'Where 55.6 GB actually goes.',body:'The dense feed-forward is 62% of the model — 34.2 GB of SwiGLU matrices that no routing can skip. Flipping to the FP8 build shrinks those matrices and the mixers, and leaves the 5.09 GB of vocabulary and the 0.92 GB vision tower exactly as they were.'},
 cache:{title:'Two kinds of memory, one stack.',body:'Sixteen layers keep a KV cache: 4 KB per token each, 17.2 GB at 262K and 65.5 GB at 1M. Forty-eight layers keep a constant recurrent state: 3.1 MB each, 151 MB total, flat in sequence length. An all-attention stack of 64 layers would need 68.7 GB at 262K just for KV.'}
};
const BENCH=[['GPQA Diamond','Reasoning',[89.2,93.4,94.1,92.9,88.1,92.4,89.9,90.9]],['Terminal-Bench 2.1','Agentic',[73,89.1,88.8,88.3,88.2,87.9,82.7,90.6]],['Terminal-Bench 3.0','Agentic',[null,43.3,34.4,17.7,28.3,11.8,7.6,30]],['Terminal-Bench 4.0','Agentic',[null,51.8,39.9,12.6,37.9,12.4,7,31.2]],['DeepSWE v1.1','Agentic',[42.2,74,73,67.5,66.9,62.7,54.4,74.2]],['ProgramBench','Agentic',[null,37,23,17.5,19,15.5,null,20.3]],['NL2Repo-Bench','Agentic',[42.3,75.3,56.8,58,58,61.5,54.2,64]],['CyberGym','Agentic',[null,null,84.5,80,84.5,83.3,76.7,88.1]],['SEC-Bench Pro','Agentic',[null,null,74.3,null,null,56.4,30.9,62.8]],['ExploitGym','Agentic',[null,22.1,33.7,null,15,5.4,1.8,15.3]],['HLE with tools','Agentic',[null,63.6,null,59.8,62.5,60,51.5,63.9]],['AutomationBench','Agentic',[null,50.3,45.8,46.7,48.8,43.2,37.7,54.8]],['Agent\\'s Last Exam','Agentic',[28.6,26.7,27.6,28.5,25.7,25.2,31.8]],['Chartography with tools','Visual',[null,84,79.9,68.1,null,null,null,78.9]],['BabyVision with tools','Visual',[85.6,94.1,88.9,85.7,null,null,null,89.6]],['ZeroBench-main (Pass@5)','Visual',[null,52,53,41,null,null,null,49]]];
const BENCH_MODELS=['Qwen3.8 27B (this atlas)','Opus-5.0','GPT-5.6 Sol','K3','GLM-5.3','DS V4 Pro','DS V4 Flash','DS V4.1 Flash']; const CARD_BENCH={"src": "Qwen/Qwen3.8-27B", "headers": ["Benchmark", "Qwen3.8-27B", "Qwen3.6-27B", "Qwen3.7-Plus", "Muse Glimmer-30B", "Opus4.6 Max"], "rows": [["== Agentic Multimodal Intelligence", "", "", "", "", ""], ["Computer use OSWorld-Verified", "84.3", "63.9", "73.3", "65.9", "72.7"], ["Browser use WebArena-Verified", "64.8", "48.8", "55.3", "--", "--"], ["Mobile use AndroidWorld", "81.9", "70.3", "81.0", "--", "62.0"], ["Application recreation RecreationBench", "47.1", "29.8", "30.2", "--", "--"], ["Multimodal tool use ClawEval-MM", "Pass@3 57.4 Average 56.9", "Pass@3 42.6 Average 50.4", "Pass@3 57.4 Average 60.1", "--", "Pass@3 52.5 Average 54.7"], ["Multimodal software engineering SWE-MM", "38.6", "25.7", "30.0", "--", "27.1"], ["Visual web development Vision2Web", "62.9", "45.0", "42.1", "--", "--"], ["== General Multimodal Intelligence", "", "", "", "", ""], ["Visual math problem solving MathVision", "Without CI 90.0 With CI 94.6", "Without CI 85.1", "Without CI 90.3", "--", "Without CI 65.5"], ["General visual reasoning BabyVision", "Without CI 65.7 With CI 85.6", "Without CI 28.9", "Without CI 64.7 With CI 70.4", "--", "Without CI 12.6"], ["Scientific chart analysis CharXiv (RQ)", "Without CI 83.7 With CI 90.2", "Without CI 78.4", "Without CI 85.8 With CI 85.9", "78.8", "Without CI 66.0"], ["Real-world perception RealWorldQA", "85.9", "84.1", "86.9", "--", "73.9"], ["Embodied intelligence ERQA", "65.5", "62.5", "69.8", "--", "40.8"]]};
/* Publisher-reported leaderboard as published on the original atlas page (each model's
   own card). Qwen3.8-27B is not in that publisher set; its own card figures are shown first for the benchmarks its card publishes, and a blank cell means the card does not publish that benchmark. */

function pickExperts(){return [];}      /* dense model: no routing to animate */
const PHASES=[
 {name:'Embed',label:'Embed',from:0,to:3,active:'5 120-dim',color:COL.enc,caption:'248,320 tokens become vectors.',desc:'A lookup in the 2.54 GB token table puts a 5,120-dimensional vector on the residual stream. Images enter through the 27-block vision tower and the merger, which projects 4,608 -> 5,120 so they join the same stream.'},
 {name:'DeltaNet x3',label:'3x DeltaNet',from:3,to:9,active:'O(1)/token',color:COL.dec,caption:'Three recurrences, no cache.',desc:'Three Gated DeltaNet layers read and write a constant 128x128 state per value head. Their cost does not grow with the context, which is why 48 of the 64 layers are this kind.'},
 {name:'Gated attention',label:'Attention',from:9,to:13,active:'4 KB/token',color:COL.attn,caption:'Every fourth layer attends the window.',desc:'A full softmax attention layer over the whole window, 24 Q heads against 4 KV heads. It writes 4 KB of KV per token — the only per-token memory in the stack.'},
 {name:'Repeat x16',label:'x16 blocks',from:13,to:18,active:'64 layers',color:COL.full,caption:'The rhythm repeats to layer 63.',desc:'Sixteen times the same pattern: three recurrences then one attention layer, each followed by the dense 17,408-wide SwiGLU feed-forward that runs in full for every token.'},
 {name:'Head + MTP',label:'Head + MTP',from:18,to:22,active:'248,320',color:COL.engram,caption:'Read out, and draft ahead.',desc:'A final RMSNorm and the untied unembedding produce 248,320 logits, while the 424.7M-parameter MTP block drafts the token after next for speculative decoding.'}
];
const TRAIN_PHASES=[
 {name:'Forward',from:0,to:8,active:'Read W',color:COL.enc,caption:'Use the weights. Predict the next tokens.',desc:'Conceptual training pass: known tokens feed the causal backbone, attention and the dense feed-forward build predictions in parallel under a causal mask. Teaching schematic, not a reproduction of Qwen training recipe.'},
 {name:'Loss',from:8,to:11,active:'Compare',color:COL.engram,caption:'Compare with the target tokens.',desc:'MTP is trained with multiple prediction steps, so the objective is not one next-token loss but a stack of them. Values here are illustrative of shape, not of the actual mixture.'},
 {name:'Backward',from:11,to:18,active:'Gradients',color:COL.mhc,caption:'Trace the error back.',desc:'Gradients follow the forward operations. Extra optimizer memory is not part of the weight solids on this page.'}
];
function phasesFor(mode){return mode==='training'?TRAIN_PHASES:PHASES;}
function phaseAt(t,mode='inference'){const ps=phasesFor(mode);return ps.find(p=>t>=p.from&&t<p.to)||ps[ps.length-1];}
const TC={q:'#8caaff',local:'#ffc477',kv:'#61dccb',index:'#f5a078',out:'#b2bfff',expert:'#66deb0',shared:'#eee081',router:'#f494be',mhc:'#c9a3fa',norm:'#90a9b5',engram:'#d7b57b',vision:'#62d5d0',head:'#a5b9eb',dnet:'#55d7c1',ffn:'#76b9ff',attn:'#55ddd0',mtp:'#b99bff',embed:'#5ca7ff',vmerge:'#9be86a'};
function tensorKind(w){return TC[w.cat]?w.cat:'head';}
function tensorColor(w){return TC[tensorKind(w)]||COL.attn;}
function tensorShort(w){return w.name.replace(/^model\\.language_model\\./,'').replace(/^model\\.visual\\./,'ViT.').replace(/^mtp\\./,'mtp.').replace(/^layers\\.\\d+\\./,'').replace(/\\.weight$/,'').replace(/^linear_attn\\./,'dnet.');}
function weightParts(w,mode){const total=wBytes(w,mode);return {data:total,scales:0,aux:0,total};}
function displayWeights(m){if(m.id!=='vision')return m.ws;const groups=new Map;for(const w of m.ws){let key=w.name.replace(/visual\\.blocks\\.\\d+\\./,'visual.blocks.*.');if(groups.has(key)){let g=groups.get(key);g.count+=w.count;g.p+=w.p;g.ex={bf16:g.ex.bf16+w.ex.bf16,fp8:g.ex.fp8+w.ex.fp8};}else groups.set(key,{...w,ex:{...w.ex}});}return [...groups.values()];}
function findWeight(m,name){return displayWeights(m).find(w=>w.name===name)||m.ws.find(w=>w.name===name);}
function orderWeights(m){const ws=displayWeights(m);const rank=w=>{let n=w.name;
 if(n.includes('input_layernorm'))return 0; if(n.startsWith('linear_attn.conv1d'))return 1;
 if(n.includes('in_proj_qkv'))return 2; if(n.includes('in_proj_z'))return 3;
 if(n.includes('in_proj_b'))return 4; if(n.includes('in_proj_a'))return 5;
 if(n.includes('A_log'))return 6; if(n.includes('dt_bias'))return 7;
 if(n.includes('linear_attn.norm'))return 8; if(n.includes('out_proj'))return 9;
 if(n.includes('q_proj'))return 1; if(n.includes('q_norm'))return 2; if(n.includes('k_proj'))return 3;
 if(n.includes('k_norm'))return 4; if(n.includes('v_proj'))return 5; if(n.includes('o_proj'))return 6;
 if(n.includes('post_attention_layernorm'))return 10; if(n.includes('gate_proj'))return 11;
 if(n.includes('up_proj'))return 12; if(n.includes('down_proj'))return 13;
 if(n.includes('patch_embed'))return 0; if(n.includes('pos_embed'))return 1;
 if(n.includes('blocks.*.attn'))return 2; if(n.includes('blocks.*.mlp'))return 3;
 if(n.includes('blocks.*.norm'))return 4; if(n.includes('merger'))return 5;
 if(n==='mtp.fc.weight')return 0; if(n.includes('pre_fc_norm'))return 1; if(n.includes('mtp.norm'))return 9;
 if(n.includes('_proj')||n.includes('q_norm'))return 3; return 8;};
 return [...ws].sort((a,b)=>rank(a)-rank(b));}
function isAttentionTensor(w){return w.cat==='attn'||w.cat==='dnet'||w.name.includes('norm');}
function tensorInfo(w,m){
 const p=fmtP(w.p),size=bytes(wBytes(w,'bf16'));
 let t='FP8 / 128x128 block-scale quantized tensor.';
 let b='Stored as e4m3 with one F32 scale per 128x128 block, so the on-disk size is 1.00024 bytes per parameter — the +0.024% is the scale overhead this page includes.';
 if(w.name.startsWith('model.language_model.embed_tokens')){t='Token table.';b='248,320 rows x 5,120. Untied from the head, shared with the MTP block, and kept BF16 by the FP8 build: every token passes through it, so it keeps the precision budget.';}
 else if(w.name.startsWith('lm_head')){t='Unembedding.';b='5,120 -> 248,320 logits, the second vocabulary-sized tensor in the checkpoint. BF16 in both builds — logit precision is where quantization noise hurts most directly.';}
 else if(w.name.includes('mtp.fc')||w.name.includes('pre_fc_norm')){t='MTP fusion and its pre-norms.';b='10,240 -> 5,120 after the pre-normed draft embedding is concatenated with the trunk hidden state. Kept BF16 by the FP8 build.';}
 else if(w.name.includes('mtp')){t='MTP block tensor.';b='The extra full-attention layer and its dense feed-forward that draft the next-next token. Its projections are quantized in the FP8 build, its norms are not.';}
 else if(w.name.includes('visual')){t='Vision tower tensor.';b='Part of the 27-block ViT: patch embedding, position table, block attention/MLP or the 4,608 -> 5,120 merger. The whole tower is on the FP8 exclude list and stays BF16.';}
 else if(w.name.includes('A_log')){t='Learned base decay.';b='48 scalars — each value head resting decay rate in log space, merged with the alpha projection every token. Highest leverage per byte in the checkpoint, and left BF16 by the FP8 build.';}
 else if(w.name.includes('dt_bias')){t='Timestep bias.';b='48 scalars biasing each head timestep; the recurrence runs in F32 (mamba_ssm_dtype: float32). BF16 on disk in both builds.';}
 else if(w.name.includes('conv1d')){t='Causal convolution, kernel 4.';b='Depthwise over the fused 10,240-dim qkv stream: each channel sees itself and the three previous tokens. 80 KB, BF16 in both builds.';}
 else if(w.name.includes('in_proj_qkv')){t='Fused QKV projection.';b='5,120 -> 10,240 in one matmul: 16 key heads x 128 + queries in the same 2,048-dim space + 48 value heads x 128. The 3:1 value-to-key asymmetry is the DeltaNet signature.';}
 else if(w.name.includes('in_proj_z')){t='Swish output gate.';b='5,120 -> 6,144: decides what the recurrence is allowed to write back onto the residual stream.';}
 else if(w.name.includes('in_proj_a')||w.name.includes('in_proj_b')){t='Decay / write gate projection.';b='5,120 -> 48: one scalar per value head per token driving how fast the state forgets (alpha) or how strongly the delta rule writes (beta). Tiny, so BF16 in both builds.';}
 else if(w.name.includes('linear_attn.norm')){t='Per-head state norm.';b='128 scales normalizing the recurrence output at head granularity.';}
 else if(w.name.includes('out_proj')){t='DeltaNet output projection.';b='Folds the gated 6,144-dim recurrence output back into the 5,120-dim residual stream.';}
 else if(w.name.includes('q_proj')){t='Query projection with a fused gate.';b='5,120 -> 12,288: twice 24 heads x 256, because the per-head swish output gate rides inside the Q projection (attn_output_gate: true). That is why no separate gate tensor exists.';}
 else if(w.name.includes('k_proj')||w.name.includes('v_proj')){t='KV projection (GQA).';b='5,120 -> 1,024: 4 KV heads x 256 dims serving all 24 query heads. Only 64 of the 256 dims carry rotary position (partial factor 0.25). Together they cost 4 KB of KV per token in these layers.';}
 else if(w.name.includes('q_norm')||w.name.includes('k_norm')){t='Per-head QK-norm.';b='256 scales — RMSNorm applied per head to queries and keys before the dot product. BF16 in both builds.';}
 else if(w.name.includes('o_proj')){t='Attention output projection.';b='6,144 -> 5,120: folds 24 gated head outputs back onto the residual stream.';}
 else if(w.name.includes('gate_proj')||w.name.includes('up_proj')){t='SwiGLU gate / up.';b='5,120 -> 17,408 — the wide half of this model parameter budget. silu(gate) x up, no routing, all 17,408 rows run for every token of every layer.';}
 else if(w.name.includes('down_proj')){t='SwiGLU down.';b='17,408 -> 5,120, returning the wide activation to the residual stream. 178.26 MB per layer at BF16, and one of the tensors the official FP8 build halves.';}
 else if(w.name.includes('norm')){t='RMSNorm.';b='Layer normalization on the residual stream (eps 1e-6). Small, load-bearing, and BF16 in both builds.';}
 return {title:t,body:b,size,p};}
function layerStory(m){
 if(!m.mode)return null;let t,body,detail;
 if(m.mode==='reuse'){t='Gated DeltaNet recurrence';body='No KV cache. A fixed 128 x (128x128) F32 state absorbs the sequence.';detail='48 value heads each keep a 128x128 state, updated by a delta rule with a learned decay and a data-dependent write gate. Cost per token is O(1) and the layer memory is ~3.1 MB whether the context holds 10 tokens or 262,144.';}
 else {t='Gated attention, full window';body='24 Q heads over 4 KV heads — and the only per-token memory in the stack.';detail='4 KB of KV per token per layer, 4 KB x 16 layers x 262,144 tokens = 17.2 GB at the native context, 65.5 GB at the 1M extension. q_proj is double width because the per-head output gate is fused into it.';}
 return {title:t,body,detail};}
"""

TAIL_ASSERT = """const VOLUME_UNIT=1e9;"""


def build_data_js(t: dict) -> str:
    lin_items = [(n, v[0], v[1], v[2]) for n, v in sorted(t["lin"].items())]
    full_items = [(n, v[0], v[1], v[2]) for n, v in sorted(t["full"].items())]
    ffn_keys = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "post_attention_layernorm.weight")
    # a linear block = its mixers; the FFN is shared by both archetypes
    lin_mixer = [x for x in lin_items if x[0] not in ffn_keys]
    ffn = [x for x in lin_items if x[0] in ffn_keys]
    full_mixer = [x for x in full_items if x[0] not in ffn_keys]
    vis = [(g[0], g[1]["b16"], g[1]["b8"], g[1]["dims"], g[1]["count"]) for g in t["vis"]]
    io = t["io"]
    tables = "\n".join([
        js_weight_array("DNET_W", lin_mixer, "lin", "one Gated DeltaNet mixer: 10 tensors"),
        js_weight_array("GATT_W", full_mixer, "full", "one gated-attention mixer: 7 tensors (q_proj is 2x wide)"),
        js_weight_array("FFN_W", ffn, "lin", "the dense SwiGLU FFN that closes every block"),
        "const LIN_W=[...DNET_W,...FFN_W];\nconst FULL_W=[...GATT_W,...FFN_W];",
        js_weight_array("EMBED_W", list(io["embed"]), "embed", "vocabulary table"),
        js_weight_array("HEAD_W", list(io["head"]), "head", "final norm + untied unembedding"),
        js_weight_array("MTP_W", list(io["mtp"]), "mtp", "MTP: fusion fc + pre-norms + one full-attention block"),
        js_weight_array("VISION_W", vis, "vis", "vision tower, 333 tensors aggregated into their logical parts"),
    ])
    cfg_fp8 = load(DIR / "hf-config-fp8.json")
    tables += ("\n/* the official FP8 build's own config.json (verbatim) — modules_to_not_convert\n"
               "   is the list that decides what this page keeps at BF16: 882 entries */\n"
               "const FP8_CONFIG=" + json.dumps(cfg_fp8, ensure_ascii=False, separators=(",", ":")) + ";\n")
    return DATA_JS.replace("@@TABLES@@", tables)


# ─────────────────── the model-specific panels (whole-line rewrites) ───────────────────
# The engine's JSX is one block per line inside its methods, so each entry replaces a single
# line of src.html. Only names that are already in scope at that point are used.
LINE_FIXES: list[tuple[int, str, str]] = [
    (775, """  ['KV cache',m.mode==='full'?'Own cache':'Absent',m.mode==='full'?'4 KB per token — 4 KV heads × 256 dims × keys and values × 2 bytes.':'A Gated DeltaNet layer appends nothing to a cache.',TC.kv,'own'],""", "ownership kv"),
    (776, """  ['Recurrent state',m.mode==='reuse'?'Own · constant':'Absent',m.mode==='reuse'?'48 value heads × 128×128 F32 rewritten in place — 3.08 MB, whatever the sequence length.':'Only the linear layers own a state.',TC.local,'own'],""", "ownership state"),
    (777, """  ['Context cost',m.mode==='full'?'Grows with context':'Flat',m.mode==='full'?'These 16 layers are the only per-token memory in the model: 17.2 GB at 262K.':'The state is overwritten, never appended to.',TC.index,'own'],""", "ownership cost"),
    (781, """  ${isLayer&&html`<${Section} title="Inside one block" tag="DENSE · NO ROUTING"><div class="expert-equation">gate = SiLU(W1 x)<br/>up = W3 x<br/>out = W2 (gate ⊙ up)</div><p class="fine">Every block closes with a 5,120 → 17,408 → 5,120 SwiGLU. There is no router, no expert bank and nothing to skip: all 17,408 rows run for every token of every layer — 534.8 MB per block at BF16, 34.2 GB model-wide.</p><${Fact} label="FFN parameters / block" value=${fmtP(3*5120*17408/1e6)}/><${Fact} label="Share of the checkpoint" value="62%"/><//>`}""", "dense ffn section"),
    (782, "  ", "drop engram section"),
    (783, """  ${m.id==='vision'&&html`<${Section} title="Images meet the language model"><div class="expert-equation">16 × 16 patches · 2 frames<br/>↓ 27 vision blocks · hidden 1,152<br/>↓ 2 × 2 spatial merge → 4,608<br/>↓ merger → 5,120 = the LM hidden size</div><p class="fine">921.5 MB of the checkpoint, and BF16 in both published builds: patch embedding 3.5 MB, a 2,304-position table 5.3 MB, 27 blocks at ~30.5 MB and an 89.7 MB merger. After the merger an image patch is an ordinary 5,120-dimensional token on the same residual stream — which is why one stack is enough.</p><//>`}""", "vision section"),
    (790, """ <${Section} title="Start with a linear layer"><p class="guide-copy">Open L00. Three of every four layers look like this: a Gated DeltaNet mixer that reads and writes one 128×128 state per value head, then the dense SwiGLU feed-forward. Nothing here grows with the context — the block is 766.5 MB on disk and 3.08 MB of state whether you send ten tokens or a hundred thousand.</p><//>""", "guide linear"),
    (791, """ <${Section} title="Then an attention layer"><p class="guide-copy">Now open L03 — one of the sixteen layers that attend the whole window. The recurrence is replaced by 24 query heads over 4 KV heads at head_dim 256, which is what costs 4 KB of cache per token. Its q_proj is 12,288 wide, twice what 24 × 256 needs, because the per-head swish output gate is fused into the Q projection.</p><//>""", "guide attention"),
    (792, """ <${Section} title="Find where the weight is"><p class="guide-copy">In any block the three mlp.* matrices are 534.8 MB of 766.5 MB — 70% of it. That ratio is the story of this model: sixty-four dense feed-forwards, none of them routable, holding 34.2 GB of the 55.6 GB checkpoint, all of it read for every token.</p><//>""", "guide ffn"),
    (793, """ <${Section} title="One memory, two prices"><p class="guide-copy">The sixteen attention layers carry a KV cache that reaches 17.2 GB at 262K and 65.5 GB at the 1M extension. The other forty-eight layers together carry 151 MB of state, flat. An all-attention stack of 64 layers would need 68.7 GB at 262K just for KV — the 3:1 rhythm is the difference between one node and none.</p><//>""", "guide memory"),
    (794, """ <${Section} title="Three distinctions worth keeping"><div class="layer-story"><h3>Dense is not small.</h3><p>No experts means no sparsity to exploit at inference: every parameter of the feed-forward is read for every token, which is exactly why the official FP8 build matters here.</p><p>State is not cache.</p><p>A Gated DeltaNet state is overwritten in place; a KV cache is appended to. Only the second one grows with your conversation.</p><p>Measured is not derived.</p><p>Every number on this page is a byte length from the published safetensors headers — asserted against those repositories' own payload totals, not a shape × bytes estimate.</p></div><//>""", "guide distinctions"),
    (797, """ <${Note} color=${s.precision==='fp8'?COL.dec:COL.enc}><strong>${EXP[s.precision].title}</strong> ${EXP[s.precision].body}<//>""", "storage note"),
    (799, """ <${Section} title="Why FP8 does not halve it"><div class="equation"><span>BF16</span><code>2.00000 B / parameter</code><span>FP8 · e4m3 · 128×128 blocks</span><code>1.00024 B / parameter ≈ 1 B</code><span>LEFT BF16 BY THE OFFICIAL BUILD</span><code>embeddings · lm_head · vision · mtp.fc · norms</code></div><p class="fine">Qwen's FP8 build converts the trunk's big matmuls only, so the checkpoint lands at 30.87 GB — 1.80× smaller, not 4×. The 5.09 GB of vocabulary tables and the 0.92 GB vision tower are byte-identical in both builds, and this page charges each tensor the size it really has, block scales included.</p><//>""", "fp8 equation"),
    (800, """ <${Section} title="Audit, rather than guess"><p class="fine">Both repositories were read the same way: each shard's safetensors header over one HTTP Range request (8-byte length prefix + JSON of dtype, shape and data_offsets), about 2.5 MB per shard, then summed per tensor across 18 and 66 shards. Reproduce it from your own copy with the header audit.</p><button class="outline-btn full-width" onClick=${()=>this.patch({modal:'audit'})}>Check exact bytes from local checkpoint headers</button><//>""", "audit note"),
    (801, """ <details class="assumptions"><summary>Accounting assumptions and limits</summary><p>Safetensors headers give the exact payload per tensor; file padding, repository extras and runtime buffers are not included. Each 128×128 block scale is folded into the tensor it scales, which is why the FP8 figures are 1.00024× the parameter count rather than exactly 1×. KV-cache and recurrent-state numbers are arithmetic on the config (4 KB per token per attention layer; 48 × 128 × 128 F32 per linear layer), not checkpoint bytes. Nothing on this page is device VRAM.</p></details></div>`;}""", "assumptions"),
    (802, """ renderCache(){const s=this.state,n=s.context,kv=n*16*4096/1e9,state=48*48*128*128*4/1e9;return html`<div><div class="inspect-path">TWO KINDS OF MEMORY</div><h2 class="editorial">Sixteen caches.<br/><em>Forty-eight states.</em></h2><p class="lede">Only the gated-attention layers grow with the conversation. Drag the context and watch which number moves. """, "cache head"),
    (803, """ <${Section} title="Stretch the context" tag="TO 1,000,000 TOKENS"><label class="range-title" for="context-number">Context tokens <input id="context-number" type="number" min="1024" max="1000000" step="1024" value=${n} onInput=${e=>this.patch({context:clamp(+e.target.value||1024,1024,1000000)})}/></label><input class="slider" type="range" min="1024" max="1000000" step="1024" value=${n} onInput=${e=>this.patch({context:+e.target.value})}/><div class="range-ends"><span>1 K</span><span>262 144 · native</span><span>1 000 000 · extended</span></div><${Fact} label="KV cache · 16 layers" value=${bytes(kv*1e9,s.binary)} color=${COL.attn}/><${Fact} label="Recurrent state · 48 layers" value=${bytes(state*1e9,s.binary)} color=${COL.dec}/><${Fact} label="If all 64 layers attended" value=${bytes(n*64*4096/1e9*1e9,s.binary)} color=${COL.muted}/>""", "cache slider"),
    (804, """ <${Section} title="Where 4 KB comes from"><div class="equation"><span>KV HEADS</span><code>4 × 256 dims = 1,024</code><span>KEYS + VALUES, 2 BYTES EACH</span><code>1,024 × 2 × 2 = 4,096 B</code><span>PER TOKEN, PER ATTENTION LAYER</span><code>16 × 262,144 × 4 KB = 17.2 GB</code></div><p class="fine">Four KV heads at head_dim 256 shared by all 24 query heads — 6:1 grouped-query attention. Only 64 of those 256 dims carry rotary position, but all of them are cached: the partial rotary saves arithmetic, not memory.</p><//>""", "kv arithmetic"),
    (805, """ <${Section} title="The other forty-eight layers"><${Fact} label="State per layer" value="3.08 MB · 48 × 128 × 128 F32"/><${Fact} label="All 48 together" value="151 MB · constant"/><${Fact} label="Grows with context?" value="No — rewritten in place"/><${Fact} label="KV cache here" value="None"/><p class="fine">A Gated DeltaNet layer read-modify-writes one 128×128 state per value head with a delta rule, steered by a learned decay and a data-dependent write gate. Ten tokens or a million: the layer holds the same 3.08 MB. Sixteen caches against forty-eight states is why a 262K context fits at all.</p><//></div>`}""", "state section"),
    (806, """ renderBench(){const s=this.state,b=BENCH[s.bench];return html`<div><div class="inspect-path">REPORTED EVALUATIONS / NATIVE MODEL</div><h2 class="editorial">Capability,<br/><em>with context.</em></h2><p class="lede">The supplied model-card results at maximum reasoning effort. No synthetic benchmark projections.</p><label class="field-label">Compare a benchmark<select value=${s.bench} onChange=${e=>this.patch({bench:Number(e.target.value)})}>${BENCH.map((r,i)=>html`<option value=${i}>${r[0]}</option>`)}</select></label><div class="benchmark-chart">${BENCH_MODELS.map((name,i)=>html`<div class=${'benchmark-row'+(i===0?' featured':'')}><div><span>${name}</span><b>${b[2][i]===null?'Not reported':b[2][i].toFixed(1)}</b></div><div class="bench-track"><i style=${{width:(b[2][i]||0)+'%',background:i===0?COL.dec:COL.enc,opacity:i===0?1:.32}}/></div></div>`)}</div><p class="fine">${b[0]} / ${b[1]}. Percentages or the card's 0-100 score; ZeroBench uses Pass@5, ProgramBench Almost@1, DeepSWE resolved rate. Do not average unlike metrics.</p><div class="card-bench"><div class="inspect-path">PUBLISHER CARD / THIS CHECKPOINT</div><p class="card-lede">Every benchmark this checkpoint&rsquo;s own card reports, with the card&rsquo;s own comparison columns, reproduced unchanged. Where a row in the series chart above reads &ldquo;Not reported&rdquo;, the card does not publish that benchmark.</p><div class="card-scroll"><table class="card-table"><thead><tr>${CARD_BENCH.headers.map(h=>html`<th>${h}</th>`)}</tr></thead><tbody>${CARD_BENCH.rows.map(r=>r[0].startsWith("== ")?html`<tr class="sec"><td colSpan=${CARD_BENCH.headers.length}>${r[0].slice(3)}</td></tr>`:html`<tr>${r.map((c,i)=>html`<td class=${i===0?"lbl":(i===1?"num own":"num")}>${c}</td>`)}</tr>`)}</tbody></table></div><p class="fine">Source: <a href="https://huggingface.co/${CARD_BENCH.src}" target="_blank" rel="noopener">${CARD_BENCH.src}</a> model card, retrieved 2026-09-15. Values are the publisher&rsquo;s; this page measures bytes and adds no numbers.</p></div>""", "engine panel line 806"),
    (807, """ <${Section} title="Evaluation conditions"><${Fact} label="Reasoning effort" value="100 / maximum"/><${Fact} label="Sampling" value="T 1.0 / top-p 0.95"/><${Fact} label="Code-agent context" value="1M tokens"/><${Fact} label="Visual-agent context" value="512K tokens"/><p class="fine">Harnesses vary. DeepSWE uses mini-SWE, Terminal-Bench uses the publisher harness, and several visual-agent evaluations use Claude Code. See the model card for task-specific protocols.</p><${Note} color=${COL.full}>Publisher-reported scores, not independent results generated here. No quality equivalence is claimed for the supplied FP8 conversion.<//><//>""", "engine panel line 807"),
    (808, """ <${Section} title="Reasoning is an API setting"><label class="range-title">reasoning_effort <output>${s.effort}</output></label><input class="slider" type="range" min="1" max="100" value=${s.effort} aria-label="Reasoning effort example" onInput=${e=>this.patch({effort:Number(e.target.value)})}/><pre class="small-code">${JSON.stringify({reasoning_effort:s.effort,temperature:1,top_p:.95},null,2)}</pre><p class="fine">This changes only the configuration example. It does not predict score, speed or token use. Benchmark bars remain fixed at the published effort-100 results.</p><//></div>`;}""", "engine panel line 808"),
    (809, """ renderSources(){return html`<div><div class="inspect-path">PROVENANCE / MEASURED 2026-09-14</div><h2 class="editorial">Trace every<br/><em>assumption.</em></h2><p class="lede">Every byte on this page came from the published safetensors headers of two Hugging Face repositories, read over HTTP Range — no weights were downloaded, and no figure is copied from a card. """, "sources head"),
    (811, """ <${Section} title="The official FP8 build"><p class="fine">Qwen/Qwen3.8-27B-FP8 is the vendor's own quantized checkpoint: e4m3 weights, one F32 scale per 128×128 block, dynamic activation scaling, and a modules_to_not_convert list naming everything it leaves alone — both embedding tables, lm_head, the entire 27-block vision tower, mtp.fc, the DeltaNet gates (A_log, dt_bias, conv1d, alpha, beta) and every RMSNorm and bias. Each tensor here is charged the size it actually has in that repository, block scales included.</p><//>""", "fp8 source section"),
    (814, """ <${Section} title="Interpretation rules"><p class="fine">The card calls this a 27B model; the measured checkpoint holds ${(TOTALS.bf16/2/1e9).toFixed(2)}B parameters once the two vocabulary tables, the 27-block vision tower and the MTP block are counted. The headline number is the trunk — the rest is inventory, and it is shown rather than folded in.</p><p class="fine">One scale is used everywhere: 1 cubic world unit = 1 decimal GB of payload, so the volume of a solid is the volume of what it costs on disk, whatever the tensor's own shape. Bytes are measured header sums; the KV and state figures are arithmetic on the config; nothing here is VRAM.</p><//>""", "interpretation rules"),
    (689, """function moduleExplanation(m){if(!m)return EXP.overview;if(m.mode)return EXP[m.mode==='full'?'full':'dnet'];if(m.id==='vision')return EXP.vision;if(m.id==='mtp')return EXP.mtp;if(m.id==='embed'||m.id==='head')return EXP.storage;if(m.id==='embed')return EXP.bf16;return EXP.overview;}""", "moduleExplanation"),
    (690, """function describeFormat(w){return w.note||(w.format==='fp8'?'e4m3 weights, one F32 scale per 128×128 block, activations quantized dynamically. This is what the official FP8 build stores.':'Two bytes per logical value in the native storage model — the size this tensor has in the BF16 checkpoint, and the size it keeps in the FP8 build when Qwen leaves it alone.');}""", "describeFormat"),
    (691, """function resolveAuditModule(name){let n=name.replace(/^(?:model\\.)+/,'').replace(/^language_model\\./,'');let a=n.match(/^layers\\.(\\d+)\\./);if(a)return 'L'+a[1];if(/^(visual\\.|vision\\.)/.test(n))return 'vision';if(/^mtp\\./.test(n))return 'mtp';if(/^embed_tokens\\./.test(n))return 'embed';if(/^(head\\.|lm_head\\.|norm\\.)/.test(n))return 'head';return null;}""", "resolveAuditModule"),
]

def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    t = templates()
    bf_total = load(BF_JSON)["payload_bytes"]
    f8_total = load(F8_JSON)["payload_bytes"]

    def sum_kind(items, idx):
        return sum(it[idx] for it in items)

    page_bf = (48 * sum(v[0] for v in t["lin"].values()) + 16 * sum(v[0] for v in t["full"].values())
               + sum_kind(t["io"]["embed"], 1) + sum_kind(t["io"]["head"], 1) + sum_kind(t["io"]["mtp"], 1)
               + sum(g[1]["b16"] for g in t["vis"]))
    page_f8 = (48 * sum(v[1] for v in t["lin"].values()) + 16 * sum(v[1] for v in t["full"].values())
               + sum_kind(t["io"]["embed"], 2) + sum_kind(t["io"]["head"], 2) + sum_kind(t["io"]["mtp"], 2)
               + sum(g[1]["b8"] for g in t["vis"]))
    if (page_bf, page_f8) != (bf_total, f8_total):
        print(f"page totals != measured payloads: {page_bf:,}/{bf_total:,} · {page_f8:,}/{f8_total:,}", file=sys.stderr)
        return 3

    html = src

    def sub(old: str, new: str, label: str, count: int = 1) -> None:
        nonlocal html
        n = html.count(old)
        if n != count:
            raise SystemExit(f"anchor '{label}': found {n}, expected {count}")
        html = html.replace(old, new)

    def splice(start: str, end: str, new: str, label: str) -> None:
        nonlocal html
        i = html.find(start)
        if i < 0:
            raise SystemExit(f"splice '{label}': start not found")
        if html.count(start) != 1:
            raise SystemExit(f"splice '{label}': start not unique")
        j = html.find(end, i + len(start))
        if j < 0:
            raise SystemExit(f"splice '{label}': end not found")
        html = html[:i] + new + html[j:]

    # 0. panel copy first: LINE_FIXES carries src.html line numbers, so it must run before the
    #    splices below (which change the file's line count). Each entry replaces one whole line.
    lines = html.split("\n")

    def set_line(n: int, text: str, label: str) -> None:
        old = lines[n - 1]
        if not old.strip():
            raise SystemExit(f"set_line '{label}': line {n} is empty")
        lines[n - 1] = text

    for n, text, label in LINE_FIXES:
        set_line(n, text, label)
    html = "\n".join(lines)

    # 1. the whole model half: `const CFG` … the line before VOLUME_UNIT (the engine's maths)
    splice("const CFG = {", "\nconst VOLUME_UNIT=1e9;", "\n" + build_data_js(t), "model data + logic")

    # 2. precision plumbing: this page has two modes, and the baseline is BF16.
    #    The mode key hides in several spellings — the TOTALS array, the P shortcut's key list,
    #    `mode='native'` defaults and the 'bf16' scale baselines — so every form is rewritten
    #    and then the absence of the old key is asserted.
    for old, new in (("['native','nvfp4','bf16']", "MODE_KEYS"),
                     ("['native','nvfp4']", "MODE_KEYS"),
                     ("['native', 'nvfp4', 'bf16']", "MODE_KEYS"),
                     ("['nvfp4','bf16']", "MODE_KEYS"),
                     ("'native'", "'bf16'"),
                     ("'nvfp4'", "'fp8'"),
                     ('"native"', "'bf16'")):
        html = html.replace(old, new)
    sub("(keys.indexOf(this.state.precision)+1)%3", "(keys.indexOf(this.state.precision)+1)%keys.length",
        "p-key modulo")

    # 4. the model-specific panels were already rewritten in step 0 (LINE_FIXES).

    # 5. copy: titles, hero, sources, exports, precision notices
    sub("<title>DeepSeek V4.1 Flash - Tensor Atlas v4</title>",
        "<title>Qwen3.8-27B - Tensor Atlas</title>", "title")
    sub('content="Offline Preact and WebGL2 architecture atlas for DeepSeek V4.1 Flash: causal encoder-decoder, CSA2, Engram, DSpark, native FP4 and supplied NVFP4 recipe, byte-proportional solids and local Safetensors auditing."',
        'content="Offline Preact and WebGL2 architecture atlas for Qwen3.8-27B: 64 dense layers — 48 Gated DeltaNet recurrences and 16 gated-attention layers, each closed by a 17,408-wide SwiGLU FFN — a 27-block vision tower and an MTP head. BF16 and the official FP8 build, byte-proportional solids measured from the safetensors headers, and local Safetensors auditing."',
        "meta description")
    sub("DeepSeek Model Atlas requires JavaScript.", "Qwen3.8-27B Tensor Atlas requires JavaScript.", "noscript")
    sub("DEEPSEEK MODEL ATLAS", "QWEN3.8-27B TENSOR ATLAS", "modal eyebrow")
    sub("<h1>DeepSeek <em>V4.1 Flash</em><span class=\"title-dot\">.</span></h1>",
        "<h1>Qwen3.8 <em>27B</em><span class=\"title-dot\">.</span></h1>", "hero")
    sub("'deepseek-v41-'+mode+'-derived-ledger.json'", "'qwen38-27b-'+mode+'-measured-ledger.json'", "ledger name")
    sub("'DeepSeek V4.1 Flash - schema-derived payload ledger'", "'Qwen3.8-27B - measured payload ledger'", "ledger title")
    sub("'deepseek-v41-3d-'+this.state.view+'.png'", "'qwen38-27b-3d-'+this.state.view+'.png'", "snapshot name")
    sub("// ---------- data.js ----------", "// ---------- data.js ----------", "data marker", 1)

    # engine provenance, stated in the file
    html = ("<!--\n  Qwen3.8-27B — Tensor Atlas\n"
            "  Engine and design: nisten/LLMViz-DeepSeek-V4.1-Flash (MIT, © 2026 netsin) — offline Preact +\n"
            "  WebGL2 shell, view tabs, transport, inspector, storage ledger, audit and CSS reused; the model\n"
            "  half (config, weight tables, categories, explanations, benchmarks, phases, copy) is Qwen3.8-27B's own.\n"
            "  Data: every byte was measured from the safetensors headers of Qwen/Qwen3.8-27B (55.5629 GB, 18 shards,\n"
            "  1,199 tensors) and Qwen/Qwen3.8-27B-FP8 (30.8667 GB, 66 shards, 1,606 tensors) over HTTP Range.\n"
            "  Metadata only — no weight data is bundled. Model licence Apache-2.0; engine licence MIT.\n-->\n") + html

    # 6. regex-level fixes: attribute strings, tab labels, gates that assumed DeepSeek modules
    html = re.sub(r'aria-label="DeepSeek[^"]*"', 'aria-label="Qwen3.8-27B Tensor Atlas"', html)
    html = html.replace("DeepSeek-AI's", "Qwen's").replace("DeepSeek-AI", "Qwen")
    sub("'cache','KV lab','03'", "'cache','Memory','03'", "cache tab label")
    sub("(phaseI===3||s.selected&&s.selected[0]==='D')", "false", "draft diagram gate", 1)
    sub("<b>+${bytes(NV_DELTA,s.binary)}</b> modeled vs native / smaller scale groups",
        "<b>−${bytes(-NV_DELTA,s.binary)}</b> saved by the official FP8 build · embeddings, vision, mtp.fc and every norm stay BF16",
        "format notice")
    html = re.sub(r"<button class=\$\{'rail-special'\+\(s\.selected==='D0'\?' selected':''\)\}[\s\S]{0,600}?</button>",
                  "", html, count=1)
    sub("NV_CONFIG", "FP8_CONFIG", "fp8 config ref", 3)

    # 7. residue pass: the shell still carried DeepSeek copy in the ledger export, the storage
    #    fine print, the parenthetical module list and the view legends.
    sub("/* DeepSeek V4.1 Flash Atlas. Model facts from the supplied configs and\n   DeepSeek's reference source; byte totals are schema-derived, not shard-audited. */",
        "/* Qwen3.8-27B Tensor Atlas. Model facts from the published config.json of\n   Qwen/Qwen3.8-27B and its official FP8 build; every byte total is a measured header sum. */",
        "old data comment")
    sub("NVFP4 uses the user-supplied backbone-routed-expert-only recipe.",
        "FP8 is the vendor build: e4m3 weights, one F32 scale per 128×128 block, dynamic activations.",
        "ledger note 1")
    sub("NVFP4 tensor-level scale overhead is modeled as 8 bytes per expert matrix; actual exporter metadata can differ.",
        "Block-scale overhead is folded into the tensor it scales (1.00024 B/param), measured per tensor from the header.",
        "ledger note 2")
    sub("BF16 is a hypothetical two-byte-per-parameter reference, not a published artifact.",
        "BF16 is the published Qwen/Qwen3.8-27B checkpoint itself — 18 shards, 1,199 tensors.", "ledger note 3")
    sub("status:'DERIVED, NOT SHARD-AUDITED'", "status:'MEASURED FROM SAFETENSORS HEADERS'", "ledger status")
    sub("NVFP4 only changes the explicitly targeted backbone routed experts.",
        "The FP8 build changes only the trunk's big matmuls.", "storage fine print")
    sub("['P','Cycle native / NVFP4 / BF16']", "['P','Cycle BF16 / official FP8']", "help keys")
    sub("It does not reproduce the complete DeepSeek training recipe",
        "It does not reproduce the complete Qwen training recipe", "training note")
    sub("'FOUR SHARED GLOBAL BANKS','890 B / original token across the model, not per layer.'",
        "'SIXTEEN KV CACHES','4 KB per token, per attention layer — 17.2 GB at 262K.'", "cache scene label")
    sub("'TOP-512 POSITIONS PER QUERY','512 illuminated sample marks illustrate sparse selection.'",
        "'FORTY-EIGHT RECURRENT STATES','3.08 MB each, rewritten in place, flat in context length.'", "cache scene label 2")
    html = html.replace("/* no DSpark stages in this model */", "/* no extra draft stages in this model */")
    html = re.sub(r'<\$\{Section\} title="Beyond the backbone">[\s\S]{0,1200}?<//>',
                  '<${Section} title="Where the memory is"><p class="fine">Two kinds of state, one stack: sixteen attention layers hold a KV cache that grows with the context, forty-eight Gated DeltaNet layers hold a constant 128×128 state per value head. Nothing else in the model keeps any per-token memory.</p><//>',
                  html, count=1)
    sub("[['expert','Experts'],['engram','Engram'],['attn','Attention']]",
        "[['ffn','Dense FFN'],['dnet','DeltaNet'],['attn','Attention']]", "storage legend")
    html = re.sub(r"\[\['full','Full'\][\s\S]{0,180}?\]\]",
                  "[['full','Attention · KV cache'],['reuse','DeltaNet · state']]", html, count=1)
    sub("${m==='bf16'?'Native config':'Supplied NVFP4 config'}",
        "${m==='bf16'?'Qwen3.8-27B · BF16':'Qwen3.8-27B-FP8 · official'}", "config labels")
    sub("Full supplied fields. Expanded arrays are shown below. Both recipes have the same underlying architecture.",
        "Both published configurations, verbatim. The FP8 build differs only in its quantization_config — the architecture is the same checkpoint, and modules_to_not_convert lists the 882 tensors it leaves at BF16.",
        "config blurb")

    # 8. scene-level structure: DeepSeek's encoder/decoder decks, its vision→aligner link, its
    #    sparse-attention field and the DSpark draft curves all point at modules this model
    #    does not have (`pos()` returns undefined → the renderer throws on `a[0]`).
    sub("this.curve(pos('vision'),pos('aligner'),TC.vision,.5,.4,.4);this.curve(pos('aligner'),pos('embed'),TC.vision,.5,.2,.6);",
        "this.curve(pos('vision'),pos('embed'),TC.vision,.5,.4,.4);", "vision->embed link")
    sub("CT.kv_source_layer_ids.forEach((id,k)=>{const p=pos('L'+id),cp=[p[0]+3.8,p[1]+.2,-2.6],n=id<20?Math.floor(s.context/2):s.context,kv=n*356;",
        "LAYERS.filter(l=>l.mode==='full').forEach((l,k)=>{const id=l.index,p=pos('L'+id),cp=[p[0]+3.8,p[1]+.2,-2.6],n=s.context,kv=n*4096;",
        "kv bank emit")
    sub("'ENCODER / L00-L19','First 20 layers / runs in decode'",
        "'FIRST HALF / L00-L31','Eight rhythm cycles: 24 DeltaNet + 8 attention layers'", "deck label 1")
    sub("'DECODER / L20-L39','Next 20 layers / runs in decode'",
        "'SECOND HALF / L32-L63','The same rhythm again, to the LM head'", "deck label 2")
    sub("'ONE CHECKPOINT / L00-L39','Ordered layers, not physical shard offsets.'",
        "'ONE STACK / L00-L63','Ordered layers, not physical shard offsets.'", "deck label 3")
    sub("V3.add(pos('D2'),[0,1,0])", "V3.add(pos('L32'),[0,1,0])", "d2 anchor")
    sub("V3.add(pos('L36'),[4.2,0,1])", "V3.add(pos('L63'),[4.2,0,1])", "l36 anchor")
    sub("V3.add(pos('L19'),[-.4,1.5,2])", "V3.add(pos('L31'),[-.4,1.5,2])", "l19 anchor")
    sub("const boundary=V3.add(pos('L20'),[-3.2,0,1.3]);", "const boundary=V3.add(pos('L32'),[-3.2,0,1.3]);", "boundary anchor")
    sub("this.label(boundary,'L19 -> L20',", "this.label(boundary,'L31 -> L32',", "boundary label")
    sub("m.index<20?COL.enc:COL.dec,1);}", "m.mode==='full'?COL.attn:COL.dec,1);}", "rail-free layer labels")
    sub("V3.add(pos('L19'),[-2,.3,0]),V3.add(pos('L20'),[-2,0,0])",
        "V3.add(pos('L31'),[-2,.3,0]),V3.add(pos('L32'),[-2,0,0])", "mid link")
    sub("const a=V3.add(pos('L19'),[0,0,2.15]),b=V3.add(pos('L20'),[0,0,2.15])",
        "const a=V3.add(pos('L31'),[0,0,2.15]),b=V3.add(pos('L32'),[0,0,2.15])", "mid arrow")
    sub("const selected=pickExperts(Math.floor(this.clock*.4),2048,512),set=new Set(selected);",
        "const selected=[],set=new Set(selected);   /* dense model: no sparse index field */",
        "no sparse field")
    sub("for(let i=0;i<2048;i++)", "for(let i=0;i<0;i++)", "no field dots")
    sub("(id<20?'2:1 / ':'1:1 / ')", "('4 KB / ')", "cache bank labels")

    # final residue report — printed last so it reflects the assembled page
    for needle in ("nvfp4", "NVFP4", "Native config", "Engram conditional", "DSpark stage", "CSA2",
                   "DeepSeek", "890 B", "552B", "engram_layer_ids", "NV_CONFIG"):
        if needle in html:
            for m in list(re.finditer(re.escape(needle), html))[:3]:
                ln = html[: m.start()].count("\n") + 1
                print(f"  residue '{needle}' L{ln}: " + html[max(0, m.start() - 70): m.end() + 70].replace("\n", "\\n")[:170])

    # WebMCP tools: webmcp-tools.js 를 자리표시자에 인라인 주입(외부 스크립트 미사용 → CSP 원문 유지)
    _wm = DIR / "webmcp-tools.js"
    if "@@WEBMCP@@" in html:
        if not _wm.exists():
            print("RESIDUE FAIL: webmcp-tools.js missing", file=sys.stderr)
            return 9
        _code = _wm.read_text(encoding="utf-8").rstrip() + "\n"
        if "</script" in _code.lower():
            print("RESIDUE FAIL: webmcp-tools.js contains </script", file=sys.stderr)
            return 9
        html = html.replace("@@WEBMCP@@", _code)
        if "@@WEBMCP@@" in html:
            print("RESIDUE FAIL: WebMCP placeholder not fully replaced", file=sys.stderr)
            return 9
        print(f"webmcp · inlined {len(_code):,} B of tools")
    OUT.write_text(html, encoding="utf-8")

    # structural guards
    for needle in ("const CFG = {", "const MODULES=", "const TOTALS=", "function layerWeights", "const VOLUME_UNIT=1e9;"):
        n = html.count(needle)
        if n != 1:
            raise SystemExit(f"'{needle}' appears {n} times (expected 1)")
    if "engram_layer_ids" in html or "index_source_layer_ids" in html:
        raise SystemExit("deepseek-only config fields are still referenced")
    js = html[html.find(">", html.find("<script>")) + 1:html.rfind("</script>")]
    probe = Path("/tmp/qwen38-tensor-module.js")
    probe.write_text(js)
    if shutil.which("node"):
        r = subprocess.run(["node", "--check", str(probe)], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr.strip()[-600:], file=sys.stderr)
            return 4
        print(f"node --check OK ({len(js):,} B inline script)")

    print(f"wrote {OUT.name}: {len(html.encode()):,} bytes (src {len(src.encode()):,})")
    print(f"measured · BF16 {bf_total/1e9:.4f} GB (18 shards) · FP8 {f8_total/1e9:.4f} GB (66 shards) "
          f"→ {bf_total/f8_total:.3f}x")
    print(f"page renders {page_bf/1e9:.4f} GB / {page_f8/1e9:.4f} GB — page total == measured payload ✔")
    print(f"modules {len(t['lin'])} linear tensors · {len(t['full'])} full · vision {len(t['vis'])} groups")

    # the NTT internal badge + copyright footer are part of the published page
    import badge
    print("badge ·", badge.inject(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
