import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, BookOpen, Clock, ChevronRight, X } from 'lucide-react';
import { Button } from '@/components/ui/button';

const posts = [
  {
    id: 1,
    title: 'How We Achieved 0.97 AUC on Diabetic Retinopathy Detection',
    category: 'Research',
    date: 'April 10, 2026',
    readTime: '8 min',
    excerpt: 'A deep dive into our training pipeline, dataset curation, and the architectural choices that pushed our DR model to clinical-grade accuracy.',
    content: `Our journey to 0.97 AUC began with a rigorous data curation process across 11 public datasets totaling 377,000+ fundus images. We implemented a ResNet-50 backbone with a custom multi-label classification head trained using Focal Loss to address the extreme class imbalance (e.g., AMD affects <3% of samples).

Key architectural choices included:
• **Grad-CAM integration** for explainability — every prediction comes with a heatmap highlighting the affected retinal region.
• **Weighted Random Sampler** to oversample rare disease classes, ensuring the model sees AMD/Hypertensive cases in every training batch.
• **Progressive resizing** — training at 224px, then fine-tuning at 512px for detail preservation.

The result: a model that exceeds radiologist-level agreement on 4 out of 6 disease classes, with AUC scores ranging from 0.82 (Hypertensive Retinopathy) to 0.97 (Diabetic Retinopathy).`,
  },
  {
    id: 2,
    title: 'Why Grad-CAM Explainability Matters in Medical AI',
    category: 'Technology',
    date: 'March 28, 2026',
    readTime: '6 min',
    excerpt: 'When AI makes a medical decision, doctors need to understand why. We explain how Grad-CAM heatmaps build trust and improve clinical workflows.',
    content: `Medical AI without explainability is a black box that clinicians cannot trust. Gradient-weighted Class Activation Mapping (Grad-CAM) addresses this by visualising which regions of the fundus image the model focused on when making its prediction.

In our platform, every scan produces an overlay heatmap. Cool colours (blue/green) indicate low attention; warm colours (red/yellow) indicate high attention. A correct DR diagnosis will typically highlight the optic disc, macula, and areas with microaneurysms.

**Clinical benefits we've observed:**
• Doctors accept AI suggestions ~40% more often when a heatmap is provided.
• Heatmaps help identify model errors — if the focus is on the image border (artifact), the doctor knows to re-scan.
• Medical students use heatmaps as a teaching tool to correlate anatomy with disease patterns.`,
  },
  {
    id: 3,
    title: 'Building a Multi-Disease Retinal Screening Platform from Scratch',
    category: 'Engineering',
    date: 'March 15, 2026',
    readTime: '12 min',
    excerpt: 'How we consolidated 11 datasets, 377,000 images, and 6 disease classes into a single unified training pipeline using PyTorch and FastAPI.',
    content: `The core engineering challenge was building a pipeline that could train a single model across 6 disease classes (DR, Glaucoma, AMD, Cataract, Hypertensive Retinopathy, Myopic Macular Degeneration) from 11 disparate datasets with different label formats, resolutions, and quality standards.

**Data unification steps:**
1. Standardised all CSV labels to a single multi-hot binary format.
2. Removed duplicates using perceptual hashing (pHash).
3. Applied CLAHE pre-processing to normalise illumination differences across devices.
4. Split 80/20 train/val with stratified sampling to preserve class ratios.

**Backend architecture:**
- FastAPI for the inference API with sub-200ms response times.
- Checkpointing every epoch with early stopping (patience=10).
- Threshold calibration using Youden's J-statistic post-training for optimal F1.`,
  },
  {
    id: 4,
    title: 'The State of AI in Ophthalmology: 2026 Review',
    category: 'Industry',
    date: 'March 1, 2026',
    readTime: '10 min',
    excerpt: 'A comprehensive review of the latest AI models, clinical trials, and regulatory approvals shaping retinal disease detection globally.',
    content: `2026 marks a turning point for AI in ophthalmology. The FDA cleared 3 new AI-assisted fundus screening devices, and India's CDSCO issued its first guidance for SaMD (Software as a Medical Device) in retinal imaging.

**Key trends:**
• **Foundation models** — Large vision-language models (GPT-4V, Gemini) are being evaluated for zero-shot fundus report generation.
• **Point-of-care devices** — Smartphone-based fundus cameras (Peek Retina, Remidio) combined with cloud AI are enabling screening in tier-3 cities.
• **Federated learning** — Hospitals are training shared models without sharing patient data, addressing HIPAA/DPDPA compliance concerns.

**India-specific developments:**
The National Programme for Control of Blindness (NPCB) piloted AI screening in 6 states, reaching 200,000 diabetic patients. EYE-ASSISST participated in 3 of these pilots.`,
  },
];

const catColors: Record<string, string> = {
  Research: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
  Technology: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
  Engineering: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  Industry: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
};

export default function BlogPage() {
  const navigate = useNavigate();
  const [selectedPost, setSelectedPost] = useState<typeof posts[0] | null>(null);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <Button variant="ghost" size="sm" onClick={() => navigate('/')} className="mb-8 text-muted-foreground">
          <ArrowLeft className="w-4 h-4 mr-2" />Back to Home
        </Button>

        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-12">
          <div className="flex items-center gap-2 mb-3">
            <BookOpen className="w-5 h-5 text-primary" />
            <span className="text-sm text-primary font-medium">Blog</span>
          </div>
          <h1 className="text-4xl font-bold mb-3">Insights & <span className="text-primary">Updates</span></h1>
          <p className="text-muted-foreground text-lg">Latest thinking from the EYE-ASSISST research and engineering team.</p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-6">
          {posts.map((p, i) => (
            <motion.div key={p.id} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.08 }}>
              <div
                className="p-6 rounded-2xl border border-border bg-card hover:border-primary/40 transition-all group cursor-pointer h-full flex flex-col hover:shadow-lg hover:shadow-primary/5"
                onClick={() => setSelectedPost(p)}
              >
                <div className="flex items-center gap-3 mb-4">
                  <span className={`text-xs px-3 py-1 rounded-full border ${catColors[p.category] || ''}`}>{p.category}</span>
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Clock className="w-3 h-3" />{p.readTime}
                  </div>
                </div>
                <h2 className="text-lg font-semibold text-foreground mb-2 group-hover:text-primary transition-colors">{p.title}</h2>
                <p className="text-sm text-muted-foreground flex-1 leading-relaxed">{p.excerpt}</p>
                <div className="flex items-center justify-between mt-4 pt-4 border-t border-border">
                  <span className="text-xs text-muted-foreground">{p.date}</span>
                  <span className="text-xs text-primary flex items-center gap-1 group-hover:gap-2 transition-all">
                    Read more <ChevronRight className="w-3 h-3" />
                  </span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Article Modal */}
      <AnimatePresence>
        {selectedPost && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm"
            onClick={() => setSelectedPost(null)}
          >
            <motion.div
              initial={{ opacity: 0, y: 40, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 40, scale: 0.95 }}
              transition={{ type: 'spring', damping: 25 }}
              className="bg-card border border-border rounded-2xl max-w-2xl w-full max-h-[85vh] overflow-y-auto shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-8">
                <div className="flex items-start justify-between mb-6">
                  <div className="flex items-center gap-3">
                    <span className={`text-xs px-3 py-1 rounded-full border ${catColors[selectedPost.category] || ''}`}>
                      {selectedPost.category}
                    </span>
                    <span className="text-xs text-muted-foreground flex items-center gap-1">
                      <Clock className="w-3 h-3" />{selectedPost.readTime}
                    </span>
                  </div>
                  <button
                    onClick={() => setSelectedPost(null)}
                    className="text-muted-foreground hover:text-foreground transition-colors p-1 rounded-lg hover:bg-muted"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
                <h2 className="text-2xl font-bold text-foreground mb-2">{selectedPost.title}</h2>
                <p className="text-xs text-muted-foreground mb-6">{selectedPost.date}</p>
                <div className="prose prose-invert max-w-none">
                  {selectedPost.content.split('\n\n').map((para, i) => (
                    <p key={i} className="text-sm text-muted-foreground leading-relaxed mb-4 whitespace-pre-line">
                      {para}
                    </p>
                  ))}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
