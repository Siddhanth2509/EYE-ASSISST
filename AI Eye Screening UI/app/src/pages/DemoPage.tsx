import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  X, 
  Activity, 
  AlertCircle, 
  CheckCircle,
  ArrowRight,
  Eye,
  Camera
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';

// Reusing the offline demo logic from App.tsx
function generateMockResult(patientId: string, laterality: string) {
  const isNormal = Math.random() > 0.6;
  if (isNormal) {
    return {
      scan_id: `DEMO-${Date.now()}`,
      patient_id: patientId || 'DEMO-PATIENT',
      laterality: laterality,
      timestamp: new Date().toISOString(),
      status: 'pending_review',
      ai_confidence: 0.95 + (Math.random() * 0.04),
      predictions: {
        dr_severity: 0,
        amd_present: false,
        glaucoma_suspect: false,
        cataract_present: false,
      },
      heatmap_url: null,
      findings: ['Normal fundus appearance', 'No signs of diabetic retinopathy', 'Optic disc margins clear'],
      recommendation: 'Routine annual screening.'
    };
  }

  const severities = [1, 2, 3, 4];
  const severity = severities[Math.floor(Math.random() * severities.length)];
  const isAmd = Math.random() > 0.7;
  const isGlaucoma = Math.random() > 0.8;
  
  const findings = [];
  if (severity === 1) findings.push('Mild microaneurysms detected');
  else if (severity === 2) findings.push('Moderate non-proliferative changes, dot-blot hemorrhages');
  else if (severity === 3) findings.push('Severe non-proliferative changes, venous beading');
  else findings.push('Proliferative retinopathy, neovascularization present');

  if (isAmd) findings.push('Macular drusen present (AMD suspect)');
  if (isGlaucoma) findings.push('Increased cup-to-disc ratio (Glaucoma suspect)');

  return {
    scan_id: `DEMO-${Date.now()}`,
    patient_id: patientId || 'DEMO-PATIENT',
    laterality: laterality,
    timestamp: new Date().toISOString(),
    status: 'pending_review',
    ai_confidence: 0.82 + (Math.random() * 0.15),
    predictions: {
      dr_severity: severity,
      amd_present: isAmd,
      glaucoma_suspect: isGlaucoma,
      cataract_present: Math.random() > 0.85,
    },
    heatmap_url: null, // Will use CSS overlay in UI instead
    findings: findings,
    recommendation: severity >= 3 ? 'Urgent ophthalmology referral required.' : 'Routine specialist review recommended within 3 months.'
  };
}

export default function DemoPage() {
  const navigate = useNavigate();
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<any | null>(null);
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (file: File) => {
    if (!file.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    setUploadedImage(url);
    setResult(null);
  };

  const startDemoAnalysis = () => {
    if (!uploadedImage) return;
    setIsAnalyzing(true);
    setProgress(0);
    
    // Simulate AI analysis progress
    const interval = setInterval(() => {
      setProgress(p => {
        if (p >= 100) {
          clearInterval(interval);
          return 100;
        }
        return p + Math.floor(Math.random() * 15) + 5;
      });
    }, 300);

    // Complete analysis
    setTimeout(() => {
      clearInterval(interval);
      setProgress(100);
      setResult(generateMockResult('DEMO-1234', 'OD'));
      setIsAnalyzing(false);
    }, 3500);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="noise-overlay" />
      
      {/* Navbar */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-background/90 backdrop-blur-lg border-b border-border h-16 flex items-center px-6">
        <div className="flex items-center gap-2 cursor-pointer" onClick={() => navigate('/')}>
          <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center">
            <Eye className="w-5 h-5 text-primary" />
          </div>
          <span className="font-bold text-lg tracking-tight">EYE-ASSISST</span>
        </div>
        <div className="ml-4 pl-4 border-l border-border flex items-center gap-2">
          <Badge variant="secondary" className="bg-amber-500/20 text-amber-400 border-amber-500/30">
            Interactive Demo Mode
          </Badge>
        </div>
        <div className="ml-auto">
          <Button variant="outline" size="sm" onClick={() => navigate('/')}>
            Exit Demo
          </Button>
        </div>
      </nav>

      <div className="pt-24 pb-12 px-6 max-w-6xl mx-auto min-h-screen flex flex-col">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">Try the AI Screening Pipeline</h1>
          <p className="text-muted-foreground">Upload a sample fundus image to see how the analysis works in real-time.</p>
        </div>

        <div className="flex-1 grid md:grid-cols-2 gap-8">
          {/* Upload Area */}
          <Card className="medical-panel flex flex-col relative overflow-hidden">
            <CardHeader>
              <CardTitle className="text-lg">Image Input</CardTitle>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col items-center justify-center p-6">
              {!uploadedImage ? (
                <div 
                  className={`w-full flex-1 border-2 border-dashed rounded-xl flex flex-col items-center justify-center transition-all ${
                    isDragging ? 'border-primary bg-primary/5 scale-[1.02]' : 'border-border bg-muted/30 hover:bg-muted/50'
                  }`}
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={(e) => { e.preventDefault(); setIsDragging(false); if (e.dataTransfer.files?.[0]) handleImageUpload(e.dataTransfer.files[0]); }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <div className="w-16 h-16 bg-background rounded-full flex items-center justify-center shadow-lg mb-4 pointer-events-none">
                    <Camera className="w-8 h-8 text-primary" />
                  </div>
                  <p className="font-medium text-lg pointer-events-none">Drop image here or click to browse</p>
                  <p className="text-sm text-muted-foreground mt-2 pointer-events-none">Supports JPG, PNG (Max 5MB)</p>
                  <input type="file" className="hidden" ref={fileInputRef} onChange={(e) => e.target.files?.[0] && handleImageUpload(e.target.files[0])} accept="image/*" />
                </div>
              ) : (
                <div className="w-full flex-1 flex flex-col items-center relative group">
                  <div className="relative w-full h-[400px] rounded-xl overflow-hidden bg-black/50 border border-border">
                    <img src={uploadedImage} alt="Uploaded fundus" className="w-full h-full object-contain" />
                    
                    {/* Simulated Heatmap Overlay if Severe */}
                    {result && result.predictions.dr_severity > 1 && (
                      <div className="absolute inset-0 pointer-events-none opacity-50 mix-blend-screen"
                           style={{ background: 'radial-gradient(circle at 60% 40%, rgba(239,68,68,0.8) 0%, rgba(249,115,22,0.4) 20%, transparent 40%)' }} />
                    )}
                  </div>

                  {!isAnalyzing && !result && (
                    <Button size="lg" className="absolute bottom-6 btn-medical shadow-xl" onClick={startDemoAnalysis}>
                      <Activity className="w-5 h-5 mr-2" /> Run AI Analysis
                    </Button>
                  )}

                  {(result || isAnalyzing) && (
                    <Button variant="secondary" size="icon" className="absolute top-4 right-4 rounded-full bg-background/80 hover:bg-background"
                      onClick={() => { setUploadedImage(null); setResult(null); setIsAnalyzing(false); }}>
                      <X className="w-4 h-4" />
                    </Button>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Results Area */}
          <Card className="medical-panel bg-card/50">
            <CardHeader>
              <CardTitle className="text-lg">Analysis Results</CardTitle>
            </CardHeader>
            <CardContent className="h-full">
              {!isAnalyzing && !result ? (
                <div className="h-full flex flex-col items-center justify-center text-muted-foreground space-y-4 py-20">
                  <Activity className="w-12 h-12 opacity-20" />
                  <p>Upload an image and run analysis to see results</p>
                </div>
              ) : isAnalyzing ? (
                <div className="h-full flex flex-col items-center justify-center py-20 space-y-8">
                  <div className="relative">
                    <div className="w-24 h-24 rounded-full border-4 border-muted flex items-center justify-center">
                      <span className="text-2xl font-bold font-mono">{progress}%</span>
                    </div>
                    <svg className="absolute inset-0 w-24 h-24 -rotate-90">
                      <circle cx="48" cy="48" r="46" fill="none" stroke="currentColor" strokeWidth="4" strokeDasharray="289" strokeDashoffset={289 - (289 * progress) / 100} className="text-primary transition-all duration-300" />
                    </svg>
                  </div>
                  <div className="text-center space-y-2">
                    <p className="font-medium animate-pulse text-primary">Processing Image...</p>
                    <p className="text-xs text-muted-foreground">Running CNN inference pipeline</p>
                  </div>
                </div>
              ) : result && (
                <AnimatePresence>
                  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
                    {/* Header */}
                    <div className="flex items-start justify-between p-4 bg-muted/30 rounded-xl border border-border">
                      <div>
                        <p className="text-sm text-muted-foreground">Overall Status</p>
                        <div className="flex items-center gap-2 mt-1">
                          {result.predictions.dr_severity === 0 ? (
                            <><CheckCircle className="w-5 h-5 text-emerald-500" /><span className="font-bold text-emerald-500">Normal</span></>
                          ) : (
                            <><AlertCircle className="w-5 h-5 text-amber-500" /><span className="font-bold text-amber-500">Abnormal Findings</span></>
                          )}
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-sm text-muted-foreground">AI Confidence</p>
                        <p className="font-bold font-mono text-primary">{(result.ai_confidence * 100).toFixed(1)}%</p>
                      </div>
                    </div>

                    {/* Pathologies */}
                    <div className="space-y-3">
                      <h4 className="font-medium text-sm text-muted-foreground">Detected Conditions</h4>
                      
                      <div className="flex justify-between items-center p-3 rounded-lg border border-border bg-background">
                        <span className="text-sm">Diabetic Retinopathy</span>
                        <Badge variant="outline" className={
                          result.predictions.dr_severity === 0 ? 'text-emerald-400 border-emerald-400/30' :
                          result.predictions.dr_severity <= 2 ? 'text-amber-400 border-amber-400/30' : 'text-red-400 border-red-400/30'
                        }>
                          {['None', 'Mild', 'Moderate', 'Severe', 'Proliferative'][result.predictions.dr_severity]}
                        </Badge>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 rounded-lg border border-border bg-background">
                        <span className="text-sm">Glaucoma Suspect</span>
                        <Badge variant="outline" className={result.predictions.glaucoma_suspect ? 'text-amber-400 border-amber-400/30' : 'text-emerald-400 border-emerald-400/30'}>
                          {result.predictions.glaucoma_suspect ? 'Detected' : 'Negative'}
                        </Badge>
                      </div>
                    </div>

                    {/* Findings */}
                    <div className="space-y-2">
                      <h4 className="font-medium text-sm text-muted-foreground">Key Findings</h4>
                      <ul className="space-y-2">
                        {result.findings.map((finding: string, i: number) => (
                          <li key={i} className="flex items-start gap-2 text-sm">
                            <span className="mt-1 w-1.5 h-1.5 bg-primary rounded-full flex-shrink-0" />
                            {finding}
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="pt-4 flex justify-end">
                      <Button onClick={() => navigate('/login')} className="btn-medical">
                        Unlock Full Platform <ArrowRight className="w-4 h-4 ml-2" />
                      </Button>
                    </div>
                  </motion.div>
                </AnimatePresence>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
