import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, FileText, Code2, Terminal, BookOpen } from 'lucide-react';
import { Button } from '@/components/ui/button';

const endpoints = [
  { method:'POST', path:'/api/analyze', desc:'Submit a fundus image for multi-disease AI analysis with optional Grad-CAM.' },
  { method:'GET',  path:'/health',      desc:'Backend health check — confirms model loaded status.' },
  { method:'GET',  path:'/api/result/{scan_id}', desc:'Retrieve a previously analyzed scan result by ID.' },
  { method:'POST', path:'/api/report/pdf', desc:'Generate and download a clinical PDF report for a scan.' },
];

const methodColors: Record<string,string> = {
  GET:'bg-emerald-500/10 text-emerald-400',
  POST:'bg-cyan-500/10 text-cyan-400',
};

export default function DocsPage() {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <Button variant="ghost" size="sm" onClick={() => navigate('/')} className="mb-8 text-muted-foreground">
          <ArrowLeft className="w-4 h-4 mr-2" />Back to Home
        </Button>

        <motion.div initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} className="mb-12">
          <div className="flex items-center gap-2 mb-3">
            <Code2 className="w-5 h-5 text-primary" />
            <span className="text-sm text-primary font-medium">Documentation</span>
          </div>
          <h1 className="text-4xl font-bold mb-3">API <span className="text-primary">Reference</span></h1>
          <p className="text-muted-foreground text-lg">Integrate EYE-ASSISST diagnostics into your own systems.</p>
        </motion.div>

        {/* Quick Start */}
        <section className="mb-12">
          <h2 className="text-xl font-semibold text-foreground mb-4 flex items-center gap-2"><Terminal className="w-5 h-5 text-primary" />Quick Start</h2>
          <div className="bg-[#0d1117] rounded-2xl border border-border p-5 font-mono text-sm">
            <div className="text-emerald-400 mb-1"># 1. Start the backend</div>
            <div className="text-foreground mb-3">cd backend && python main.py</div>
            <div className="text-emerald-400 mb-1"># 2. Analyze an image</div>
            <div className="text-foreground">curl -X POST http://127.0.0.1:8000/api/analyze \<br/>
              &nbsp;&nbsp;-F "file=@retinal_image.jpg" \<br/>
              &nbsp;&nbsp;-F "patient_id=P-001" \<br/>
              &nbsp;&nbsp;-F "laterality=OD"</div>
          </div>
        </section>

        {/* Endpoints */}
        <section className="mb-12">
          <h2 className="text-xl font-semibold text-foreground mb-4 flex items-center gap-2"><FileText className="w-5 h-5 text-primary" />Endpoints</h2>
          <div className="space-y-3">
            {endpoints.map((ep,i) => (
              <motion.div key={ep.path} initial={{ opacity:0, x:-10 }} animate={{ opacity:1, x:0 }} transition={{ delay:i*0.07 }}
                className="flex items-start gap-4 p-4 rounded-xl border border-border bg-card">
                <span className={`text-xs font-bold font-mono px-3 py-1 rounded-lg flex-shrink-0 ${methodColors[ep.method]}`}>
                  {ep.method}
                </span>
                <div>
                  <code className="text-sm text-foreground font-mono">{ep.path}</code>
                  <p className="text-sm text-muted-foreground mt-1">{ep.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Response Example */}
        <section className="mb-12">
          <h2 className="text-xl font-semibold text-foreground mb-4 flex items-center gap-2"><BookOpen className="w-5 h-5 text-primary" />Sample Response</h2>
          <div className="bg-[#0d1117] rounded-2xl border border-border p-5 font-mono text-sm text-foreground overflow-x-auto whitespace-pre">
{`{
  "analysis_id": "SCAN-17632849",
  "timestamp": "2026-04-18T22:00:00Z",
  "dr_binary": {
    "is_dr": true,
    "confidence": 0.891
  },
  "dr_severity": {
    "grade": 2,
    "label": "Moderate DR",
    "color": "#F97316"
  },
  "multi_disease": {
    "glaucoma":     { "detected": false, "confidence": 0.12 },
    "amd":          { "detected": false, "confidence": 0.08 },
    "cataract":     { "detected": true,  "confidence": 0.64 },
    "hypertensive": { "detected": false, "confidence": 0.05 },
    "myopic":       { "detected": false, "confidence": 0.03 }
  },
  "gradcam": {
    "heatmap_base64": "<base64 PNG string>"
  }
}`}
          </div>
        </section>
      </div>
    </div>
  );
}
