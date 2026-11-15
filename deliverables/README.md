# Team Praise: final submission (Gradient Learnings Data Analytics Hackathon)

This folder holds only what gets submitted. Everything else (code, data, report sources, earlier drafts) lives outside it.

| # | Deliverable | File | What to do |
|---|---|---|---|
| 1 | **Google Colab notebook** | `1_Colab_Notebook/VoltRelay_Analysis.ipynb` | Upload to Google Drive → *Open with → Google Colaboratory* → *Share → Anyone with the link can view* → paste the link in the form (or use the *Open in Colab* badge in the repo README once the repo is public). It already contains every output, so judges can read it without running it. *Runtime → Run all* re-runs it end to end (~3 min of compute, ~10–15 min on Colab including the data download). |
| 2 | **Analysis report** | `2_Analysis_Report/VoltRelay_Analysis_Report.pdf` | 31 pages. Upload or attach the PDF (or put it on Drive and share the link). |
| 3 | **3-minute video** | `3_Video/` | Record it from the deck and script below, save the MP4 here, then upload it to Drive or YouTube (unlisted) and share the link. |

## 3_Video

| File | What it is |
|---|---|
| `VoltRelay_Video_Deck.pptx` | The 10-slide deck. Each slide's speaker notes hold its narration, starting with a time window such as `[0:47-1:12]`. |
| `VoltRelay_Video_Deck.pdf` | The same slides as a PDF, for viewing. |
| `Video_Script_3min.md` | The recording script: 424 words, about 2:55 at 145 words a minute, with a timing table, a "point to" cue per slide and delivery tips. Order follows the brief: business problem → approach → insights → recommendations. |

**To record:** open the .pptx in PowerPoint → *Slide Show → Record* → read the speaker notes →
*File → Export → Create a Video* (Full HD 1080p) → save it here as `VoltRelay_Video_TeamPraise.mp4`.
If you run long, shorten slide 3 (approach) first. After editing the deck or notes, `python deck/finalize.py` rebuilds the
deck, the PDF and the script together.

## Before you submit
- [ ] Notebook opens in Colab, and the link is set to "Anyone with the link can view"
- [ ] Report PDF attached or linked
- [ ] Video is 3:00 or shorter, exported as MP4, and the link is shared with view access
- [ ] Submitted through the form before **11:59 PM IST, 15 Nov 2026**

*Outside this folder:* `analysis/` (analysis scripts), `src/` (cleaning, notebook and report builders), `deck/` (deck
builder), `outputs/` (figures, tables, findings), `working/` (report sources, build intermediates), `data/` (Parquet copies of
the dataset, not in the repo).
