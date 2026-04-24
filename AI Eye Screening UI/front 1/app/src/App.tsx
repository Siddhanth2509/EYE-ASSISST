import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Eye,
  Upload,
  User,
  LogOut,
  Activity,
  History,
  BarChart3,
  CheckCircle,
  XCircle,
  Edit3,
  ZoomIn,
  ZoomOut,
  Maximize2,
  AlertCircle,
  FileText,
  TrendingUp,
  Users,
  Cpu,
  Shield,
  Stethoscope,
  Microscope,
  X,
  Loader2,
  ArrowLeft,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Switch } from '@/components/ui/switch';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Slider } from '@/components/ui/slider';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
} from 'recharts';

// Components
import CursorEffect from '@/components/CursorEffect';
import EyeBot from '@/components/EyeBot';
import ReportGenerator from '@/components/ReportGenerator';

// Pages
import LandingPage from '@/pages/LandingPage';
import PatientPortal from '@/pages/PatientPortal';

// Types
type UserRole = 'doctor' | 'technician' | 'admin' | 'patient' | null;

interface AnalysisResult {
  scan_id: string;
  patient_id: string;
  timestamp: string;
  binary_result: string;
  confidence: number;
  severity_level: number;
  severity_label: string;
  severity_color: string;
  multi_disease_flags: Record<string, any>;
  gradcam_available: boolean;
  original_image?: string;
  heatmap_image?: string;
}

// Mock Data
const MOCK_PATIENT_HISTORY = [
  { scan_id: 'SCAN-001', timestamp: '2026-03-20T10:30:00Z', laterality: 'OD', severity_level: 1, severity_label: 'Mild', severity_color: '#F59E0B', confidence: 87.5, review_status: 'approved' },
  { scan_id: 'SCAN-002', timestamp: '2026-02-15T14:20:00Z', laterality: 'OS', severity_level: 0, severity_label: 'Normal', severity_color: '#10B981', confidence: 94.2, review_status: 'approved' },
  { scan_id: 'SCAN-003', timestamp: '2026-01-10T09:15:00Z', laterality: 'OD', severity_level: 2, severity_label: 'Moderate', severity_color: '#F97316', confidence: 91.8, review_status: 'modified' },
  { scan_id: 'SCAN-004', timestamp: '2025-12-05T16:45:00Z', laterality: 'OS', severity_level: 1, severity_label: 'Mild', severity_color: '#F59E0B', confidence: 82.3, review_status: 'approved' },
];

const TREND_DATA = [
  { date: '2025-09', severity: 0, confidence: 92 },
  { date: '2025-10', severity: 1, confidence: 88 },
  { date: '2025-11', severity: 1, confidence: 85 },
  { date: '2025-12', severity: 1, confidence: 82 },
  { date: '2026-01', severity: 2, confidence: 91 },
  { date: '2026-02', severity: 0, confidence: 94 },
  { date: '2026-03', severity: 1, confidence: 87 },
];

const ANALYTICS_DATA = {
  dailyScans: [
    { day: 'Mon', scans: 45, approved: 38, modified: 5, overridden: 2 },
    { day: 'Tue', scans: 52, approved: 44, modified: 6, overridden: 2 },
    { day: 'Wed', scans: 48, approved: 40, modified: 6, overridden: 2 },
    { day: 'Thu', scans: 61, approved: 52, modified: 7, overridden: 2 },
    { day: 'Fri', scans: 55, approved: 47, modified: 6, overridden: 2 },
    { day: 'Sat', scans: 32, approved: 28, modified: 3, overridden: 1 },
    { day: 'Sun', scans: 28, approved: 24, modified: 3, overridden: 1 },
  ],
  severityDistribution: [
    { name: 'Normal', value: 245, color: '#10B981' },
    { name: 'Mild', value: 128, color: '#F59E0B' },
    { name: 'Moderate', value: 67, color: '#F97316' },
    { name: 'Severe', value: 34, color: '#EF4444' },
    { name: 'Proliferative', value: 12, color: '#DC2626' },
  ],
  modelPerformance: {
    sensitivity: 0.94,
    specificity: 0.89,
    auc: 0.96,
    accuracy: 0.91,
  }
};

// ============================================================================
// SVG CIRCULAR PROGRESS RING
// Accepts value in 0-1 (decimal) or 0-100 (percentage) range
// ============================================================================
function CircularProgress({ value, color, size = 60, strokeWidth = 5, label }: {
  value: number; color: string; size?: number; strokeWidth?: number; label: string;
}) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  // Normalize: if value is 0-1 (e.g., 0.12), multiply by 100. If already 0-100, leave as-is.
  const normalizedValue = value <= 1 ? value * 100 : value;
  const offset = circumference - (normalizedValue / 100) * circumference;

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={size} height={size} className="transform -rotate-90">
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke="currentColor" strokeWidth={strokeWidth}
          className="text-muted/20"
        />
        <motion.circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke={color} strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.5, ease: 'easeOut' }}
        />
      </svg>
      <span className="text-xs font-medium">{label}</span>
      <span className="text-xs text-muted-foreground">{(value * 100).toFixed(0)}%</span>
    </div>
  );
}

// ============================================================================
// HEADER
// ============================================================================
function Header({ userRole, onLogout }: { userRole: UserRole; onLogout: () => void }) {
  const [currentTime, setCurrentTime] = useState(new Date());
  const navigate = useNavigate();

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const roleLabels: Record<string, string> = {
    doctor: 'Doctor', technician: 'Technician', admin: 'Administrator', patient: 'Patient'
  };

  const roleIcons: Record<string, React.ReactNode> = {
    doctor: <Stethoscope className="w-4 h-4" />,
    technician: <Microscope className="w-4 h-4" />,
    admin: <Shield className="w-4 h-4" />,
    patient: <User className="w-4 h-4" />,
  };

  return (
    <header className="h-14 bg-card border-b border-border flex items-center justify-between px-4 lg:px-6">
      <div className="flex items-center gap-3">
        <button onClick={() => navigate('/')} className="flex items-center gap-3 hover:opacity-80 transition-opacity">
          <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center">
            <Eye className="w-5 h-5 text-primary" />
          </div>
          <span className="font-semibold text-lg tracking-tight">EYE-ASSISST</span>
        </button>
        <Badge variant="outline" className="ml-2 text-xs hidden sm:inline-flex">AI-Powered Screening</Badge>
      </div>
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="hidden sm:inline">System Online</span>
        </div>
        <Separator orientation="vertical" className="h-6 hidden sm:block" />
        {userRole && (
          <div className="flex items-center gap-2">
            {roleIcons[userRole]}
            <span className="text-sm font-medium hidden sm:inline">{roleLabels[userRole]}</span>
          </div>
        )}
        <Separator orientation="vertical" className="h-6 hidden sm:block" />
        <span className="text-sm text-muted-foreground mono hidden md:inline">
          {currentTime.toLocaleTimeString('en-US', { hour12: false })} UTC
        </span>
        <Button variant="ghost" size="icon" onClick={onLogout} className="h-8 w-8">
          <LogOut className="w-4 h-4" />
        </Button>
      </div>
    </header>
  );
}

// ============================================================================
// LOGIN PAGE
// ============================================================================
function LoginPage({ onLogin }: { onLogin: (role: UserRole) => void }) {
  const [selectedRole, setSelectedRole] = useState<UserRole>(null);
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = () => {
    if (!selectedRole) return;
    setIsLoading(true);
    setTimeout(() => {
      onLogin(selectedRole);
      setIsLoading(false);
      if (selectedRole === 'patient') {
        navigate('/patient');
      }
    }, 800);
  };

  const roles = [
    { id: 'doctor', label: 'Doctor', icon: Stethoscope, description: 'Review AI analyses and approve reports' },
    { id: 'technician', label: 'Technician', icon: Microscope, description: 'Upload images and run AI screening' },
    { id: 'admin', label: 'Administrator', icon: Shield, description: 'Monitor system metrics and performance' },
    { id: 'patient', label: 'Patient', icon: User, description: 'View your reports and book appointments' },
  ];

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="w-full max-w-md">
        <div className="text-center mb-8">
          <motion.div initial={{ scale: 0.8 }} animate={{ scale: 1 }} transition={{ delay: 0.2, type: 'spring' }} className="w-20 h-20 bg-primary/10 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <Eye className="w-10 h-10 text-primary" />
          </motion.div>
          <h1 className="text-3xl font-bold mb-2">EYE-ASSISST</h1>
          <p className="text-muted-foreground">AI-Powered Eye Disease Screening</p>
        </div>

        <Card className="medical-panel">
          <CardHeader>
            <CardTitle className="text-lg">Select Your Role</CardTitle>
            <CardDescription>Choose your role to access the appropriate dashboard</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {roles.map((role) => (
              <button
                key={role.id}
                onClick={() => setSelectedRole(role.id as UserRole)}
                className={`w-full flex items-start gap-4 p-4 rounded-lg border cursor-pointer transition-all text-left ${
                  selectedRole === role.id ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50 hover:bg-muted/50'
                }`}
              >
                <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center flex-shrink-0 mt-0.5 ${
                  selectedRole === role.id ? 'border-primary' : 'border-muted-foreground'
                }`}>
                  {selectedRole === role.id && <div className="w-2.5 h-2.5 bg-primary rounded-full" />}
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <role.icon className="w-4 h-4 text-primary" />
                    <span className="font-medium">{role.label}</span>
                  </div>
                  <p className="text-sm text-muted-foreground mt-1">{role.description}</p>
                </div>
              </button>
            ))}

            <Button className="w-full btn-medical mt-4" onClick={handleLogin} disabled={isLoading || !selectedRole}>
              {isLoading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Accessing...</> : <><User className="w-4 h-4 mr-2" />Access Dashboard</>}
            </Button>

            <Button variant="ghost" className="w-full" onClick={() => navigate('/')}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Home
            </Button>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}

// ============================================================================
// TECHNICIAN DASHBOARD (Enhanced with Grad-CAM Slider & SVG Rings)
// ============================================================================
function TechnicianDashboard() {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [patientId, setPatientId] = useState('');
  const [laterality, setLaterality] = useState('OD');
  const [heatmapOpacity, setHeatmapOpacity] = useState([50]);
  const [showReport, setShowReport] = useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFile(file);
  };

  const handleFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      setUploadedImage(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleAnalyze = async () => {
    if (!uploadedImage || !patientId) return;
    setIsAnalyzing(true);
    await new Promise(resolve => setTimeout(resolve, 2500));

    const mockResult: AnalysisResult = {
      scan_id: `SCAN-${Math.random().toString(36).substr(2, 8).toUpperCase()}`,
      patient_id: patientId,
      timestamp: new Date().toISOString(),
      binary_result: Math.random() > 0.3 ? 'DR Detected' : 'Normal',
      confidence: Math.round(85 + Math.random() * 12),
      severity_level: Math.floor(Math.random() * 3),
      severity_label: ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative'][Math.floor(Math.random() * 3)],
      severity_color: ['#10B981', '#F59E0B', '#F97316', '#EF4444', '#DC2626'][Math.floor(Math.random() * 3)],
      multi_disease_flags: {
        amd: { detected: false, confidence: 0.12 },
        glaucoma: { detected: false, confidence: 0.28 },
        cataract: { detected: true, confidence: 0.41 },
      },
      gradcam_available: true,
      original_image: uploadedImage,
      heatmap_image: uploadedImage,
    };

    setAnalysisResult(mockResult);
    setIsAnalyzing(false);
  };

  return (
    <div className="flex h-[calc(100vh-56px)]">
      {/* Left Panel */}
      <motion.div initial={{ x: -40, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ duration: 0.6 }}
        className="w-80 bg-card border-r border-border flex flex-col">
        <ScrollArea className="flex-1">
          <div className="p-4 space-y-6">
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">Image Upload</h3>
              <div onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }} onDragLeave={() => setIsDragging(false)} onDrop={handleDrop}
                className={`drop-zone p-6 text-center cursor-pointer ${isDragging ? 'drag-over' : ''}`}>
                <input type="file" accept="image/*" onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} className="hidden" id="file-upload" />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Upload className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">Drag & drop fundus image</p>
                  <p className="text-xs text-muted-foreground mt-1">or click to browse</p>
                </label>
              </div>
            </div>

            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">Patient Information</h3>
              <div className="space-y-3">
                <div>
                  <Label className="text-xs">Patient ID</Label>
                  <Input placeholder="e.g., P-2026-0041" value={patientId} onChange={(e) => setPatientId(e.target.value)} className="mt-1" />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label className="text-xs">Age</Label>
                    <Input type="number" placeholder="Years" className="mt-1" />
                  </div>
                  <div>
                    <Label className="text-xs">Laterality</Label>
                    <div className="flex gap-2 mt-1">
                      <Button variant={laterality === 'OD' ? 'default' : 'outline'} size="sm" onClick={() => setLaterality('OD')} className="flex-1">OD</Button>
                      <Button variant={laterality === 'OS' ? 'default' : 'outline'} size="sm" onClick={() => setLaterality('OS')} className="flex-1">OS</Button>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Button className="w-full btn-medical" onClick={handleAnalyze} disabled={isAnalyzing || !uploadedImage}>
                {isAnalyzing ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Analyzing...</> : <><Activity className="w-4 h-4 mr-2" />Analyze Image</>}
              </Button>
              <Button variant="outline" className="w-full" onClick={() => { setUploadedImage(null); setAnalysisResult(null); }}>
                <X className="w-4 h-4 mr-2" />Reset
              </Button>
            </div>

            {/* History */}
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">Prior Scans</h3>
              <div className="space-y-2">
                {MOCK_PATIENT_HISTORY.slice(0, 3).map((scan) => (
                  <div key={scan.scan_id} className="p-3 rounded-lg border border-border hover:bg-muted/50 cursor-pointer transition-colors">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">{scan.scan_id}</span>
                      <Badge variant="outline" className="text-xs" style={{ borderColor: scan.severity_color, color: scan.severity_color }}>{scan.severity_label}</Badge>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">{new Date(scan.timestamp).toLocaleDateString()} • {scan.laterality}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </ScrollArea>
      </motion.div>

      {/* Center Canvas */}
      <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.5, delay: 0.2 }}
        className="flex-1 bg-[#070A12] flex flex-col">
        {isAnalyzing && <div className="h-1 bg-border overflow-hidden"><div className="h-full progress-sweep" /></div>}
        <div className="flex-1 flex items-center justify-center p-8">
          {uploadedImage ? (
            <div className="relative corner-brackets max-w-2xl w-full">
              <div className="relative">
                <img src={uploadedImage} alt="Fundus" className="w-full object-contain border border-white/10 rounded-lg" />
                {analysisResult && (
                  <div className="absolute inset-0 pointer-events-none" style={{ opacity: heatmapOpacity[0] / 100 }}>
                    <div className="absolute inset-0 bg-gradient-to-br from-orange-500/40 via-red-500/30 to-transparent rounded-lg" />
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="text-center text-muted-foreground">
              <Eye className="w-16 h-16 mx-auto mb-4 opacity-30" />
              <p>No image loaded. Upload to begin.</p>
            </div>
          )}
        </div>
      </motion.div>

      {/* Right Panel - Enhanced Results */}
      <AnimatePresence>
        {analysisResult && (
          <motion.div initial={{ x: 40, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 40, opacity: 0 }} transition={{ duration: 0.5 }}
            className="w-80 bg-card border-l border-border overflow-auto">
            <div className="p-4 space-y-4">
              {/* Severity Banner */}
              <motion.div initial={{ y: -20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.2 }}
                className="p-3 rounded-lg text-center" style={{ backgroundColor: `${analysisResult.severity_color}15`, border: `1px solid ${analysisResult.severity_color}30` }}>
                <span className="text-sm font-bold" style={{ color: analysisResult.severity_color }}>
                  {analysisResult.severity_label} DR Detected
                </span>
              </motion.div>

              {/* Grad-CAM Slider */}
              <div>
                <Label className="text-xs flex justify-between">
                  <span>Original</span>
                  <span>Heatmap Preview (Demo)</span>
                </Label>
                <Slider value={heatmapOpacity} onValueChange={setHeatmapOpacity} max={100} step={1} className="mt-2" />
                <p className="text-xs text-muted-foreground mt-1 text-center">{heatmapOpacity[0]}% heatmap overlay</p>
              </div>

              {/* Confidence */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-muted-foreground">Confidence</span>
                  <span className="text-lg font-bold mono">{analysisResult.confidence}%</span>
                </div>
                <Progress value={analysisResult.confidence} className="h-2" />
              </div>

              {/* Disease Rings */}
              <div>
                <h4 className="text-xs font-semibold uppercase text-muted-foreground mb-3">Disease Screening</h4>
                <div className="flex justify-around">
                  {Object.entries(analysisResult.multi_disease_flags).map(([disease, data]: [string, any]) => (
                    <CircularProgress key={disease} value={data.confidence} color={data.detected ? '#F59E0B' : '#10B981'} label={disease.toUpperCase()} />
                  ))}
                </div>
              </div>

              {/* Summary */}
              <div className="text-sm space-y-1">
                <p className="text-muted-foreground text-xs">Key findings:</p>
                <ul className="list-disc list-inside text-xs space-y-1">
                  <li>Microaneurysms detected in temporal region</li>
                  <li>Hemorrhage probability elevated</li>
                  <li>Macula appears within normal limits</li>
                </ul>
              </div>

              {/* Generate Report Button */}
              <motion.div
                initial={{ opacity: 0.6, scale: 0.97 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.5, duration: 0.8, ease: 'easeOut' }}
              >
                <Button className="w-full btn-medical" onClick={() => setShowReport(true)}>
                  <FileText className="w-4 h-4 mr-2" />
                  Generate Clinical Report
                </Button>
              </motion.div>

              {/* Report Modal */}
              {showReport && (
                <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
                  <div className="w-full max-w-4xl max-h-[90vh] overflow-auto bg-background rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold">Clinical Report</h3>
                      <Button variant="ghost" size="icon" onClick={() => setShowReport(false)}><X className="w-4 h-4" /></Button>
                    </div>
                    <ReportGenerator data={analysisResult as any} onClose={() => setShowReport(false)} />
                  </div>
                </div>
              )}

              <div className="text-xs text-muted-foreground space-y-1">
                <p>Scan: <span className="mono">{analysisResult.scan_id}</span></p>
                <p>Time: {new Date(analysisResult.timestamp).toLocaleString()}</p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ============================================================================
// DOCTOR REVIEW PORTAL
// ============================================================================
function DoctorReviewPortal() {
  const [selectedScan, setSelectedScan] = useState<AnalysisResult | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [reviewAction, setReviewAction] = useState('');
  const [notes, setNotes] = useState('');
  const [zoom, setZoom] = useState(100);
  const [showReport, setShowReport] = useState(false);

  const pendingScans: AnalysisResult[] = [
    { scan_id: 'SCAN-A7B2C9D1', patient_id: 'P-2026-0042', timestamp: '2026-03-25T09:30:00Z', binary_result: 'DR Detected', confidence: 91.5, severity_level: 2, severity_label: 'Moderate', severity_color: '#F97316', multi_disease_flags: {}, gradcam_available: true },
    { scan_id: 'SCAN-E8F3G4H5', patient_id: 'P-2026-0043', timestamp: '2026-03-25T10:15:00Z', binary_result: 'Normal', confidence: 96.2, severity_level: 0, severity_label: 'Normal', severity_color: '#10B981', multi_disease_flags: {}, gradcam_available: true },
  ];

  const handleSubmitReview = () => {
    setSelectedScan(null);
    setReviewAction('');
    setNotes('');
  };

  return (
    <div className="flex h-[calc(100vh-56px)]">
      <motion.div initial={{ x: -40, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ duration: 0.6 }} className="w-72 bg-card border-r border-border">
        <ScrollArea className="h-full">
          <div className="p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-4">Pending Reviews ({pendingScans.length})</h3>
            <div className="space-y-2">
              {pendingScans.map((scan) => (
                <div key={scan.scan_id} onClick={() => setSelectedScan(scan)}
                  className={`p-3 rounded-lg border cursor-pointer transition-all ${selectedScan?.scan_id === scan.scan_id ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50 hover:bg-muted/50'}`}>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{scan.patient_id}</span>
                    <Badge variant="outline" className="text-xs" style={{ borderColor: scan.severity_color, color: scan.severity_color }}>{scan.severity_label}</Badge>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">{scan.scan_id}</p>
                </div>
              ))}
            </div>
          </div>
        </ScrollArea>
      </motion.div>

      <div className="flex-1 bg-[#070A12] flex flex-col">
        {selectedScan ? (
          <>
            <div className="h-12 border-b border-border flex items-center justify-between px-4">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <Switch checked={showHeatmap} onCheckedChange={setShowHeatmap} id="heatmap-toggle" />
                  <Label htmlFor="heatmap-toggle" className="text-sm">Heatmap</Label>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setZoom(Math.max(50, zoom - 25))}><ZoomOut className="w-4 h-4" /></Button>
                <span className="text-xs mono w-12 text-center">{zoom}%</span>
                <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setZoom(Math.min(200, zoom + 25))}><ZoomIn className="w-4 h-4" /></Button>
              </div>
            </div>
            <div className="flex-1 flex">
              <div className="flex-1 flex items-center justify-center p-4 border-r border-border">
                <div className="text-center">
                  <p className="text-xs text-muted-foreground mb-2">Original Image</p>
                  <div className="corner-brackets w-64 h-64 bg-muted rounded-lg flex items-center justify-center">
                    <Eye className="w-16 h-16 opacity-30" />
                  </div>
                </div>
              </div>
              <div className="flex-1 flex items-center justify-center p-4">
                <div className="text-center">
                  <p className="text-xs text-muted-foreground mb-2">{showHeatmap ? 'Grad-CAM Heatmap' : 'Original'}</p>
                  <div className="corner-brackets w-64 h-64 bg-muted rounded-lg flex items-center justify-center relative">
                    <Eye className="w-16 h-16 opacity-30" />
                    {showHeatmap && <div className="absolute inset-0 bg-gradient-to-br from-orange-500/50 via-red-500/30 to-transparent rounded-lg" />}
                  </div>
                </div>
              </div>
            </div>
            <div className="h-12 border-t border-border flex items-center justify-center gap-4 px-4">
              <span className="text-xs text-muted-foreground">Model Attention:</span>
              <div className="w-48 heatmap-legend" />
              <span className="text-xs text-muted-foreground">High</span>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <FileText className="w-16 h-16 mx-auto mb-4 opacity-30" />
              <p>Select a scan to review</p>
            </div>
          </div>
        )}
      </div>

      <AnimatePresence>
        {selectedScan && (
          <motion.div initial={{ x: 40, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 40, opacity: 0 }} transition={{ duration: 0.5 }} className="w-80 bg-card border-l border-border overflow-auto">
            <div className="p-4 space-y-4">
              <Card className="border-primary/30">
                <CardHeader className="pb-3"><CardTitle className="text-sm">AI Assessment</CardTitle></CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Result</span>
                    <Badge variant="outline" style={{ borderColor: selectedScan.severity_color, color: selectedScan.severity_color }}>{selectedScan.severity_label}</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Confidence</span>
                    <span className="font-bold mono">{selectedScan.confidence}%</span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3"><CardTitle className="text-sm">Doctor Review</CardTitle></CardHeader>
                <CardContent className="space-y-4">
                  <RadioGroup value={reviewAction} onValueChange={setReviewAction}>
                    <div className="space-y-2">
                      <Label htmlFor="approve" className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${reviewAction === 'approve' ? 'border-emerald-500 bg-emerald-500/10' : 'border-border'}`}>
                        <RadioGroupItem value="approve" id="approve" />
                        <CheckCircle className="w-4 h-4 text-emerald-500" />
                        <span className="text-sm">Approve AI result</span>
                      </Label>
                      <Label htmlFor="modify" className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${reviewAction === 'modify' ? 'border-amber-500 bg-amber-500/10' : 'border-border'}`}>
                        <RadioGroupItem value="modify" id="modify" />
                        <Edit3 className="w-4 h-4 text-amber-500" />
                        <span className="text-sm">Modify diagnosis</span>
                      </Label>
                      <Label htmlFor="override" className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${reviewAction === 'override' ? 'border-red-500 bg-red-500/10' : 'border-border'}`}>
                        <RadioGroupItem value="override" id="override" />
                        <XCircle className="w-4 h-4 text-red-500" />
                        <span className="text-sm">Reject / Re-scan</span>
                      </Label>
                    </div>
                  </RadioGroup>
                  <div>
                    <Label className="text-sm">Clinical Notes</Label>
                    <Textarea value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Add your clinical observations..." className="mt-1 min-h-[100px]" />
                  </div>
                  <Button className="w-full btn-medical" onClick={handleSubmitReview}>
                    <CheckCircle className="w-4 h-4 mr-2" />Submit Review
                  </Button>
                  <Button variant="outline" className="w-full" onClick={() => setShowReport(true)}>
                    <FileText className="w-4 h-4 mr-2" />Generate Report
                  </Button>
                </CardContent>
              </Card>

              {showReport && (
                <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
                  <div className="w-full max-w-4xl max-h-[90vh] overflow-auto bg-background rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold">Clinical Report</h3>
                      <Button variant="ghost" size="icon" onClick={() => setShowReport(false)}><X className="w-4 h-4" /></Button>
                    </div>
                    <ReportGenerator data={selectedScan as any} onClose={() => setShowReport(false)} />
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ============================================================================
// PATIENT HISTORY DASHBOARD
// ============================================================================
function PatientHistoryDashboard() {
  const [selectedPatient, setSelectedPatient] = useState('P-2026-0041');
  const [compareMode, setCompareMode] = useState(false);
  const [selectedScans, setSelectedScans] = useState<string[]>([]);

  const patients = [
    { id: 'P-2026-0041', name: 'John Doe', age: 58, scans: 7 },
    { id: 'P-2026-0042', name: 'Jane Smith', age: 65, scans: 4 },
    { id: 'P-2026-0043', name: 'Robert Johnson', age: 72, scans: 12 },
  ];

  const toggleScanSelection = (scanId: string) => {
    if (selectedScans.includes(scanId)) setSelectedScans(selectedScans.filter(id => id !== scanId));
    else if (selectedScans.length < 2) setSelectedScans([...selectedScans, scanId]);
  };

  return (
    <div className="flex h-[calc(100vh-56px)]">
      <motion.div initial={{ x: -40, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ duration: 0.6 }} className="w-72 bg-card border-r border-border">
        <ScrollArea className="h-full">
          <div className="p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Patients</h3>
              <Badge variant="outline">{patients.length}</Badge>
            </div>
            <div className="space-y-2">
              {patients.map((patient) => (
                <div key={patient.id} onClick={() => setSelectedPatient(patient.id)}
                  className={`p-3 rounded-lg border cursor-pointer transition-all ${selectedPatient === patient.id ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50 hover:bg-muted/50'}`}>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{patient.name}</span>
                    <Badge variant="secondary" className="text-xs">{patient.scans} scans</Badge>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">{patient.id} • {patient.age} years</p>
                </div>
              ))}
            </div>
          </div>
        </ScrollArea>
      </motion.div>

      <div className="flex-1 overflow-auto">
        <div className="p-6 space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold">{patients.find(p => p.id === selectedPatient)?.name}</h2>
              <p className="text-muted-foreground">{selectedPatient} • {patients.find(p => p.id === selectedPatient)?.age} years</p>
            </div>
            <Button variant={compareMode ? 'default' : 'outline'} onClick={() => setCompareMode(!compareMode)}>
              <Maximize2 className="w-4 h-4 mr-2" />{compareMode ? 'Exit Compare' : 'Compare Scans'}
            </Button>
          </div>

          <Card className="medical-panel">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-primary" />Severity Trend Over Time
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={TREND_DATA}>
                    <defs>
                      <linearGradient id="colorSeverity" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#22D3EE" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#22D3EE" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="date" stroke="rgba(255,255,255,0.5)" fontSize={12} />
                    <YAxis stroke="rgba(255,255,255,0.5)" fontSize={12} domain={[0, 4]} ticks={[0, 1, 2, 3, 4]} />
                    <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                    <Area type="monotone" dataKey="severity" stroke="#22D3EE" fillOpacity={1} fill="url(#colorSeverity)" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <div>
            <h3 className="text-lg font-semibold mb-4">Scan History</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {MOCK_PATIENT_HISTORY.map((scan) => (
                <Card key={scan.scan_id} className={`cursor-pointer transition-all ${compareMode && selectedScans.includes(scan.scan_id) ? 'border-primary ring-2 ring-primary/20' : ''}`}
                  onClick={() => compareMode && toggleScanSelection(scan.scan_id)}>
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">{scan.scan_id}</span>
                      {compareMode && (
                        <div className={`w-5 h-5 rounded border flex items-center justify-center ${selectedScans.includes(scan.scan_id) ? 'bg-primary border-primary' : 'border-border'}`}>
                          {selectedScans.includes(scan.scan_id) && <CheckCircle className="w-3 h-3" />}
                        </div>
                      )}
                    </div>
                    <div className="flex items-center gap-2 mb-2">
                      <Badge variant="outline" style={{ borderColor: scan.severity_color, color: scan.severity_color }}>{scan.severity_label}</Badge>
                      <span className="text-xs text-muted-foreground">{scan.laterality}</span>
                    </div>
                    <p className="text-xs text-muted-foreground">{new Date(scan.timestamp).toLocaleDateString()}</p>
                    <div className="mt-2 flex items-center gap-2">
                      <Progress value={scan.confidence} className="h-1.5 flex-1" />
                      <span className="text-xs mono">{scan.confidence}%</span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// ADMIN ANALYTICS PANEL
// ============================================================================
function AdminAnalyticsPanel() {
  return (
    <div className="h-[calc(100vh-56px)] overflow-auto">
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold">Analytics Dashboard</h2>
            <p className="text-muted-foreground">System performance and screening statistics</p>
          </div>
          <Badge variant="outline" className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />System Online
          </Badge>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            { icon: Activity, label: 'Total Scans', value: '486', change: '+12%', color: 'text-emerald-400' },
            { icon: CheckCircle, label: 'Model Accuracy', value: '91%', detail: 'AUC: 96%', color: 'text-emerald-400' },
            { icon: AlertCircle, label: 'Override Rate', value: '4.2%', detail: '20 of 476', color: 'text-amber-400' },
            { icon: Users, label: 'Active Users', value: '12', detail: '3 doctors, 8 techs', color: 'text-blue-400' },
          ].map((stat) => (
            <Card key={stat.label} className="medical-panel">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">{stat.label}</p>
                    <p className="text-3xl font-bold mono">{stat.value}</p>
                  </div>
                  <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
                    <stat.icon className="w-6 h-6 text-primary" />
                  </div>
                </div>
                <p className={`text-xs mt-2 ${stat.color}`}>{stat.change || stat.detail}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="medical-panel">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2"><BarChart3 className="w-5 h-5 text-primary" />Daily Screening Volume</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={ANALYTICS_DATA.dailyScans}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="day" stroke="rgba(255,255,255,0.5)" fontSize={12} />
                    <YAxis stroke="rgba(255,255,255,0.5)" fontSize={12} />
                    <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                    <Bar dataKey="scans" fill="#22D3EE" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card className="medical-panel">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2"><Activity className="w-5 h-5 text-primary" />Severity Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={ANALYTICS_DATA.severityDistribution} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value">
                      {ANALYTICS_DATA.severityDistribution.map((entry, index) => <Cell key={`cell-${index}`} fill={entry.color} />)}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </div>

        <Card className="medical-panel">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2"><Cpu className="w-5 h-5 text-primary" />Model Performance Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {[
                { label: 'Sensitivity', value: ANALYTICS_DATA.modelPerformance.sensitivity, color: 'text-emerald-400' },
                { label: 'Specificity', value: ANALYTICS_DATA.modelPerformance.specificity, color: 'text-blue-400' },
                { label: 'AUC-ROC', value: ANALYTICS_DATA.modelPerformance.auc, color: 'text-primary' },
                { label: 'Accuracy', value: ANALYTICS_DATA.modelPerformance.accuracy, color: 'text-amber-400' },
              ].map((metric) => (
                <div key={metric.label} className="text-center">
                  <p className="text-sm text-muted-foreground mb-1">{metric.label}</p>
                  <p className={`text-4xl font-bold mono ${metric.color}`}>{(metric.value * 100).toFixed(0)}%</p>
                  <Progress value={metric.value * 100} className="h-2 mt-2" />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// ============================================================================
// MAIN APP COMPONENT
// ============================================================================
function AppContent() {
  const [userRole, setUserRole] = useState<UserRole>(null);
  const [activeTab, setActiveTab] = useState('upload');
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogin = (role: UserRole) => {
    setUserRole(role);
    if (role === 'doctor') setActiveTab('review');
    else if (role === 'technician') setActiveTab('upload');
    else if (role === 'admin') setActiveTab('analytics');
  };

  const handleLogout = () => {
    setUserRole(null);
    setActiveTab('upload');
    navigate('/login');
  };

  const navItems: Record<string, { id: string; label: string; icon: React.ComponentType<{ className?: string }> }[]> = {
    doctor: [
      { id: 'review', label: 'Review Portal', icon: Stethoscope },
      { id: 'history', label: 'Patient History', icon: History },
    ],
    technician: [
      { id: 'upload', label: 'Image Upload', icon: Upload },
      { id: 'history', label: 'Patient History', icon: History },
    ],
    admin: [
      { id: 'analytics', label: 'Analytics', icon: BarChart3 },
      { id: 'history', label: 'Patient History', icon: History },
    ],
  };

  return (
    <div className="min-h-screen bg-background">
      <CursorEffect />

      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<LoginPage onLogin={handleLogin} />} />
        <Route path="/patient" element={<PatientPortal />} />
        <Route path="/dashboard" element={
          userRole ? (
            <>
              <Header userRole={userRole} onLogout={handleLogout} />
              <div className="flex h-[calc(100vh-56px)]">
                <motion.aside initial={{ x: -20, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ duration: 0.4 }}
                  className="w-16 bg-card border-r border-border flex flex-col items-center py-4 gap-2">
                  {navItems[userRole]?.map((item) => (
                    <button key={item.id} onClick={() => setActiveTab(item.id)}
                      className={`w-12 h-12 rounded-lg flex items-center justify-center transition-all ${activeTab === item.id ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-muted hover:text-foreground'}`}
                      title={item.label}>
                      <item.icon className="w-5 h-5" />
                    </button>
                  ))}
                </motion.aside>
                <main className="flex-1 overflow-hidden">
                  <AnimatePresence mode="wait">
                    <motion.div key={activeTab} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.3 }} className="h-full">
                      {activeTab === 'upload' && <TechnicianDashboard />}
                      {activeTab === 'review' && <DoctorReviewPortal />}
                      {activeTab === 'history' && <PatientHistoryDashboard />}
                      {activeTab === 'analytics' && <AdminAnalyticsPanel />}
                    </motion.div>
                  </AnimatePresence>
                </main>
              </div>
              <EyeBot />
            </>
          ) : (
            <LoginPage onLogin={handleLogin} />
          )
        } />
      </Routes>

      {/* Show EyeBot on all pages except login */}
      {location.pathname !== '/login' && <EyeBot />}
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}
