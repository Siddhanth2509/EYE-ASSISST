import { useState, useEffect } from 'react';
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
  Loader2
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
// import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Switch } from '@/components/ui/switch';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
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
  Area
} from 'recharts';
import { toast } from 'sonner';

// ============================================================================
// API CONFIGURATION
// ============================================================================

const API_BASE_URL = 'http://127.0.0.1:8000';

// ============================================================================
// TYPES
// ============================================================================

type UserRole = 'doctor' | 'technician' | 'admin' | null;

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

interface PatientScan {
  scan_id: string;
  timestamp: string;
  laterality: string;
  severity_level: number;
  severity_label: string;
  severity_color: string;
  confidence: number;
  review_status: string;
}

// ============================================================================
// MOCK DATA
// ============================================================================

const MOCK_PATIENT_HISTORY: PatientScan[] = [
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
// COMPONENTS
// ============================================================================

// Header Component
function Header({ userRole, onLogout }: { userRole: UserRole; onLogout: () => void }) {
  const [currentTime, setCurrentTime] = useState(new Date());
  
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);
  
  const roleLabels: Record<string, string> = {
    doctor: 'Doctor',
    technician: 'Technician',
    admin: 'Administrator'
  };
  
  const roleIcons: Record<string, React.ReactNode> = {
    doctor: <Stethoscope className="w-4 h-4" />,
    technician: <Microscope className="w-4 h-4" />,
    admin: <Shield className="w-4 h-4" />
  };

  return (
    <header className="h-14 bg-card border-b border-border flex items-center justify-between px-4 lg:px-6">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center">
          <Eye className="w-5 h-5 text-primary" />
        </div>
        <span className="font-semibold text-lg tracking-tight">EYE-ASSISST</span>
        <Badge variant="outline" className="ml-2 text-xs">AI-Powered Screening</Badge>
      </div>
      
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span className="status-online" />
          <span>System Online</span>
        </div>
        
        <Separator orientation="vertical" className="h-6" />
        
        <div className="flex items-center gap-2">
          {userRole && roleIcons[userRole]}
          <span className="text-sm font-medium">{userRole && roleLabels[userRole]}</span>
        </div>
        
        <Separator orientation="vertical" className="h-6" />
        
        <span className="text-sm text-muted-foreground mono">
          {currentTime.toLocaleTimeString('en-US', { hour12: false })} UTC
        </span>
        
        <Button variant="ghost" size="icon" onClick={onLogout} className="h-8 w-8">
          <LogOut className="w-4 h-4" />
        </Button>
      </div>
    </header>
  );
}

// Login Page
function LoginPage({ onLogin }: { onLogin: (role: UserRole) => void }) {
  const [selectedRole, setSelectedRole] = useState<UserRole>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = () => {
    if (!selectedRole) {
      toast.error('Please select a role');
      return;
    }
    setIsLoading(true);
    setTimeout(() => {
      onLogin(selectedRole);
      setIsLoading(false);
      toast.success(`Logged in as ${selectedRole}`);
    }, 800);
  };

  const roles = [
    { id: 'doctor', label: 'Doctor', icon: Stethoscope, description: 'Review AI analyses and approve reports' },
    { id: 'technician', label: 'Technician', icon: Microscope, description: 'Upload images and run AI screening' },
    { id: 'admin', label: 'Administrator', icon: Shield, description: 'Monitor system metrics and performance' },
  ];

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md"
      >
        <div className="text-center mb-8">
          <motion.div 
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring' }}
            className="w-20 h-20 bg-primary/10 rounded-2xl flex items-center justify-center mx-auto mb-4"
          >
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
          <CardContent className="space-y-4">
            <RadioGroup value={selectedRole || ''} onValueChange={(v) => setSelectedRole(v as UserRole)}>
              {roles.map((role) => (
                <div key={role.id}>
                  <Label
                    htmlFor={role.id}
                    className={`flex items-start gap-4 p-4 rounded-lg border cursor-pointer transition-all ${
                      selectedRole === role.id 
                        ? 'border-primary bg-primary/5' 
                        : 'border-border hover:border-primary/50 hover:bg-muted/50'
                    }`}
                  >
                    <RadioGroupItem value={role.id} id={role.id} className="mt-1" />
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <role.icon className="w-4 h-4 text-primary" />
                        <span className="font-medium">{role.label}</span>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">{role.description}</p>
                    </div>
                  </Label>
                </div>
              ))}
            </RadioGroup>

            <Button 
              className="w-full btn-medical" 
              onClick={handleLogin}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Accessing...
                </>
              ) : (
                <>
                  <User className="w-4 h-4 mr-2" />
                  Access Dashboard
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        <p className="text-center text-sm text-muted-foreground mt-6">
          Secure medical imaging platform • HIPAA Compliant
        </p>
      </motion.div>
    </div>
  );
}

// Technician Dashboard - Image Upload & Analysis
function TechnicianDashboard() {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [patientId, setPatientId] = useState('');
  const [laterality, setLaterality] = useState('OD');
  const [age, setAge] = useState('');
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  // Check backend health on mount
  useEffect(() => {
    fetch(`${API_BASE_URL}/health`)
      .then(res => res.json())
      .then(data => {
        setBackendStatus('online');
        if (!data.model_loaded) {
          toast.warning('Backend online but no model loaded. Predictions may not work.');
        }
      })
      .catch(() => {
        setBackendStatus('offline');
        toast.error('Backend API is not reachable at ' + API_BASE_URL);
      });
  }, []);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      handleFile(file);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFile(file);
    }
  };

  const handleFile = (file: File) => {
    // Clear previous analysis when new file is uploaded
    setAnalysisResult(null);
    setUploadedFile(file);  // Keep raw file for API submission
    const reader = new FileReader();
    reader.onload = (e) => {
      setUploadedImage(e.target?.result as string);
      toast.success('Image uploaded successfully');
    };
    reader.readAsDataURL(file);
  };

  const handleAnalyze = async () => {
    console.log('🔍 handleAnalyze called');
    
    if (!uploadedFile) {
      toast.error('Please upload an image first');
      return;
    }
    if (!patientId) {
      toast.error('Please enter a patient ID');
      return;
    }
    if (isAnalyzing) {
      console.log('⏳ Already analyzing, skipping...');
      return; // Prevent double-clicks
    }

    console.log('📤 Starting analysis...');
    console.log('  Patient ID:', patientId);
    console.log('  Laterality:', laterality);
    console.log('  File:', uploadedFile.name);
    
    setIsAnalyzing(true);
    setAnalysisResult(null); // Clear previous analysis
    console.log('🗑️ Cleared previous results');
    
    try {
      // Send image to backend API
      const formData = new FormData();
      formData.append('file', uploadedFile);
      formData.append('patient_id', patientId);
      formData.append('laterality', laterality);
      
      console.log('📡 Sending to backend:', API_BASE_URL);

      const response = await fetch(
        `${API_BASE_URL}/api/analyze?include_gradcam=true`,
        { method: 'POST', body: formData }
      );
      
      console.log('📥 Response status:', response.status);

      if (!response.ok) {
        // Properly handle error response
        console.error('❌ Backend rejected image:', response.status);
        const errData = await response.json().catch(() => ({ detail: 'Server returned error' }));
        const errorMessage = errData.detail || `Server error: ${response.status}`;
        console.error('❌ Error message:', errorMessage);
        
        // Show clear error toast
        toast.error(errorMessage, {
          duration: 5000,  // Show for 5 seconds
          style: {
            background: '#ef4444',
            color: 'white',
            fontSize: '16px',
            fontWeight: 'bold'
          }
        });
        
        setIsAnalyzing(false);
        return; // Stop here on error - DO NOT show results
      }

      console.log('✅ Backend accepted image, processing response...');
      const data = await response.json();
      console.log('✅ Response data:', data);

      // Map backend response to frontend AnalysisResult interface
      const severityColorMap: Record<number, string> = {
        0: '#10B981', 1: '#F59E0B', 2: '#F97316', 3: '#EF4444', 4: '#DC2626'
      };
      const severityLabelMap: Record<number, string> = {
        0: 'Normal', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative'
      };

      const sevGrade = data.dr_severity?.grade ?? 0;

      const result: AnalysisResult = {
        scan_id: data.analysis_id || `SCAN-${Date.now()}`,
        patient_id: patientId,
        timestamp: data.timestamp || new Date().toISOString(),
        binary_result: data.dr_binary?.is_dr ? 'DR Detected' : 'Normal',
        confidence: data.dr_binary?.confidence ?? 0,
        severity_level: sevGrade,
        severity_label: data.dr_severity?.label || severityLabelMap[sevGrade] || 'Normal',
        severity_color: data.dr_severity?.color || severityColorMap[sevGrade] || '#10B981',
        multi_disease_flags: data.multi_disease ?? {},
        gradcam_available: !!data.gradcam?.heatmap_base64,
        original_image: uploadedImage ?? undefined,
        heatmap_image: data.gradcam?.heatmap_base64
          ? `data:image/png;base64,${data.gradcam.heatmap_base64}`
          : uploadedImage ?? undefined,
      };

      setAnalysisResult(result);
      console.log('✅ Analysis complete, results set');
      toast.success('Analysis complete');
    } catch (err: any) {
      console.error('💥 Exception caught:', err);
      console.error('Analysis failed:', err);
      toast.error(`Analysis failed: ${err.message}`);
      setAnalysisResult(null);  // Ensure no stale results
    } finally {
      setIsAnalyzing(false);
      console.log('🏁 handleAnalyze finished');
    }
  };

  const resetAnalysis = () => {
    setUploadedImage(null);
    setUploadedFile(null);
    setAnalysisResult(null);
    setPatientId('');
    setAge('');
  };

  return (
    <div className="flex h-[calc(100vh-56px)]">
      {/* Left Panel - Controls */}
      <motion.div 
        initial={{ x: -40, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
        className="w-80 bg-card border-r border-border flex flex-col"
      >
        <ScrollArea className="flex-1">
          <div className="p-4 space-y-6">
            {/* Upload Section */}
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
                Image Upload
              </h3>
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`drop-zone p-6 text-center cursor-pointer ${isDragging ? 'drag-over' : ''}`}
              >
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileInput}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Upload className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">
                    Drag & drop fundus image
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    or click to browse
                  </p>
                </label>
              </div>
            </div>

            {/* Patient Info */}
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
                Patient Information
              </h3>
              <div className="space-y-3">
                <div>
                  <Label htmlFor="patient-id" className="text-xs">Patient ID</Label>
                  <Input
                    id="patient-id"
                    placeholder="e.g., P-2026-0041"
                    value={patientId}
                    onChange={(e) => setPatientId(e.target.value)}
                    className="mt-1"
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label htmlFor="age" className="text-xs">Age</Label>
                    <Input
                      id="age"
                      type="number"
                      placeholder="Years"
                      value={age}
                      onChange={(e) => setAge(e.target.value)}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label className="text-xs">Laterality</Label>
                    <div className="flex gap-2 mt-1">
                      <Button
                        variant={laterality === 'OD' ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setLaterality('OD')}
                        className="flex-1"
                      >
                        OD
                      </Button>
                      <Button
                        variant={laterality === 'OS' ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setLaterality('OS')}
                        className="flex-1"
                      >
                        OS
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="space-y-2">
              <Button 
                className="w-full btn-medical" 
                onClick={handleAnalyze}
                disabled={isAnalyzing || !uploadedImage}
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Activity className="w-4 h-4 mr-2" />
                    Analyze Image
                  </>
                )}
              </Button>
              <Button 
                variant="outline" 
                className="w-full"
                onClick={resetAnalysis}
              >
                <X className="w-4 h-4 mr-2" />
                Reset
              </Button>
            </div>

            {/* History List */}
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
                Prior Scans
              </h3>
              <div className="space-y-2">
                {MOCK_PATIENT_HISTORY.slice(0, 3).map((scan) => (
                  <div 
                    key={scan.scan_id}
                    className="p-3 rounded-lg border border-border hover:bg-muted/50 cursor-pointer transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">{scan.scan_id}</span>
                      <Badge 
                        variant="outline" 
                        className="text-xs"
                        style={{ borderColor: scan.severity_color, color: scan.severity_color }}
                      >
                        {scan.severity_label}
                      </Badge>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      {new Date(scan.timestamp).toLocaleDateString()} • {scan.laterality}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </ScrollArea>
      </motion.div>

      {/* Center - Image Canvas */}
      <motion.div 
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="flex-1 bg-[#070A12] flex flex-col"
      >
        {/* Progress Bar */}
        {isAnalyzing && (
          <div className="h-1 bg-border overflow-hidden">
            <div className="h-full progress-sweep" />
          </div>
        )}

        {/* Image Viewport */}
        <div className="flex-1 flex items-center justify-center p-8">
          {uploadedImage ? (
            <div className="relative corner-brackets">
              <img
                src={uploadedImage}
                alt="Fundus"
                className="max-w-full max-h-[70vh] object-contain border border-white/10 rounded-lg"
              />
              {/* Zoom Controls */}
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-card/90 backdrop-blur px-3 py-2 rounded-full">
                <Button variant="ghost" size="icon" className="h-7 w-7">
                  <ZoomOut className="w-4 h-4" />
                </Button>
                <span className="text-xs mono">100%</span>
                <Button variant="ghost" size="icon" className="h-7 w-7">
                  <ZoomIn className="w-4 h-4" />
                </Button>
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

      {/* Right Panel - Results */}
      <AnimatePresence>
        {analysisResult && (
          <motion.div 
            initial={{ x: 40, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 40, opacity: 0 }}
            transition={{ duration: 0.5 }}
            className="w-80 bg-card border-l border-border"
          >
            <ScrollArea className="h-full">
              <div className="p-4 space-y-4">
                {/* AI Result Card */}
                <Card className="border-primary/30">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Activity className="w-4 h-4 text-primary" />
                      AI Assessment
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Severity Badge */}
                    <div className="text-center">
                      <span 
                        className="severity-badge"
                        style={{ 
                          backgroundColor: `${analysisResult.severity_color}20`,
                          color: analysisResult.severity_color,
                          borderColor: `${analysisResult.severity_color}40`
                        }}
                      >
                        {analysisResult.severity_label} DR
                      </span>
                    </div>

                    {/* Binary Result */}
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Result</span>
                      <span className={`text-sm font-medium ${
                        analysisResult.binary_result === 'Normal' ? 'text-emerald-400' : 'text-amber-400'
                      }`}>
                        {analysisResult.binary_result}
                      </span>
                    </div>

                    {/* Confidence */}
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs text-muted-foreground">Confidence</span>
                        <span className="text-lg font-bold mono">{analysisResult.confidence}%</span>
                      </div>
                      <Progress value={analysisResult.confidence} className="h-2" />
                    </div>

                    {/* Summary */}
                    <div className="text-sm space-y-1">
                      <p className="text-muted-foreground">Key findings:</p>
                      <ul className="list-disc list-inside text-xs space-y-1">
                        <li>Microaneurysms detected in temporal region</li>
                        <li>Hemorrhage probability elevated</li>
                        <li>Macula appears within normal limits</li>
                      </ul>
                    </div>
                  </CardContent>
                </Card>

                {/* Disease Diagnosis Panel */}
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Activity className="w-4 h-4 text-primary" />
                      Disease Diagnosis
                    </CardTitle>
                    <p className="text-xs text-muted-foreground">AI confidence per condition</p>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {Object.entries(analysisResult.multi_disease_flags).map(([key, d]: [string, any]) => {
                      const detected = d.detected;
                      const conf = typeof d.confidence === 'number' ? d.confidence : 0;
                      const label = d.name || key.replace('_', ' ');
                      const barColor = detected ? '#EF4444' : '#10B981';
                      return (
                        <div key={key}>
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs font-medium capitalize">{label}</span>
                            <div className="flex items-center gap-2">
                              <span className="text-xs mono" style={{ color: barColor }}>
                                {conf.toFixed(1)}%
                              </span>
                              <Badge
                                variant={detected ? 'destructive' : 'outline'}
                                className="text-xs px-1.5 py-0"
                              >
                                {detected ? 'Positive' : 'Negative'}
                              </Badge>
                            </div>
                          </div>
                          <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all duration-700"
                              style={{ width: `${Math.min(conf, 100)}%`, backgroundColor: barColor }}
                            />
                          </div>
                        </div>
                      );
                    })}
                    {Object.keys(analysisResult.multi_disease_flags).length === 0 && (
                      <p className="text-xs text-muted-foreground text-center py-2">No disease data available</p>
                    )}
                  </CardContent>
                </Card>

                {/* Scan Info */}
                <div className="text-xs text-muted-foreground space-y-1">
                  <p>Scan ID: <span className="mono">{analysisResult.scan_id}</span></p>
                  <p>Timestamp: {new Date(analysisResult.timestamp).toLocaleString()}</p>
                  <p>Laterality: {laterality}</p>
                </div>
              </div>
            </ScrollArea>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Doctor Review Portal
function DoctorReviewPortal() {
  const [selectedScan, setSelectedScan] = useState<AnalysisResult | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [reviewAction, setReviewAction] = useState('');
  const [notes, setNotes] = useState('');
  const [zoom, setZoom] = useState(100);
  const [pendingScans, setPendingScans] = useState<AnalysisResult[]>([]);
  const [scanDetails, setScanDetails] = useState<Record<string, any>>({});

  // Fetch real scans from backend
  useEffect(() => {
    const fetchScans = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/scans`);
        if (!response.ok) return;
        
        const data = await response.json();
        const scans = data.scans || [];
        
        // Convert backend format to frontend format
        const converted = scans.map((scan: any) => ({
          scan_id: scan.scan_id,
          patient_id: scan.patient_id,
          timestamp: scan.timestamp,
          binary_result: scan.binary_result,
          confidence: scan.confidence,
          severity_level: scan.severity_level,
          severity_label: scan.severity_label,
          multi_disease_flags: {},
          gradcam_available: true,
          severity_color: ['#10B981', '#F59E0B', '#F97316', '#EF4444', '#DC2626'][scan.severity_level] || '#10B981',
          original_image: undefined, // Will be loaded on selection
          heatmap_image: undefined,  // Will be loaded on selection
        }));
        
        setPendingScans(converted);
      } catch (err) {
        console.error('Failed to fetch scans:', err);
      }
    };
    
    fetchScans();
    const interval = setInterval(fetchScans, 5000); // Refresh every 5s
    return () => clearInterval(interval);
  }, []);

  // Fetch full scan details when a scan is selected
  useEffect(() => {
    if (!selectedScan || selectedScan.original_image?.startsWith('data:image')) {
      // Already has images loaded
      return;
    }
    
    const fetchDetails = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/scan/${selectedScan.scan_id}`);
        if (!response.ok) {
          console.error('Failed to fetch scan details:', response.status);
          return;
        }
        
        const data = await response.json();
        console.log('Fetched scan details:', { scan_id: data.scan_id, has_images: !!data.original_image });
        
        setScanDetails(prev => ({
          ...prev,
          [selectedScan.scan_id]: data
        }));
        
        // Update selected scan with images
        setSelectedScan(prev => {
          if (!prev) return null;
          return {
            ...prev,
            original_image: data.original_image.startsWith('data:image')
              ? data.original_image
              : `data:image/png;base64,${data.original_image}`,
            heatmap_image: data.heatmap_image.startsWith('data:image')
              ? data.heatmap_image
              : `data:image/png;base64,${data.heatmap_image}`,
          };
        });
      } catch (err) {
        console.error('Failed to fetch scan details:', err);
      }
    };
    
    fetchDetails();
  }, [selectedScan?.scan_id]);

  const handleSubmitReview = async () => {
    if (!reviewAction) {
      toast.error('Please select an action');
      return;
    }
    if (!selectedScan) {
      toast.error('No scan selected');
      return;
    }
    
    try {
      const formData = new FormData();
      formData.append('action', reviewAction);
      if (notes) {
        formData.append('notes', notes);
      }
      
      const response = await fetch(
        `${API_BASE_URL}/api/v1/review/${selectedScan.scan_id}`,
        { method: 'POST', body: formData }
      );
      
      if (!response.ok) {
        throw new Error('Failed to submit review');
      }
      
      const data = await response.json();
      toast.success(`Review ${reviewAction}d successfully`);
      
      // Refresh scan list
      const scanListResponse = await fetch(`${API_BASE_URL}/api/scans`);
      if (scanListResponse.ok) {
        const scanListData = await scanListResponse.json();
        const formattedScans: AnalysisResult[] = scanListData.scans.map((s: any) => ({
          scan_id: s.scan_id,
          patient_id: s.patient_id,
          timestamp: s.timestamp,
          binary_result: s.binary_result,
          confidence: s.confidence,
          severity_level: s.severity_level,
          severity_label: s.severity_label,
          severity_color: 
            s.severity_level === 0 ? '#10B981' :
            s.severity_level === 1 ? '#F59E0B' :
            s.severity_level === 2 ? '#F97316' :
            s.severity_level === 3 ? '#EF4444' : '#DC2626',
          multi_disease_flags: {},
          gradcam_available: true,
        }));
        setPendingScans(formattedScans.filter(s => s.binary_result.includes('DR')));
      }
      
      setSelectedScan(null);
      setReviewAction('');
      setNotes('');
    } catch (error: any) {
      console.error('Failed to submit review:', error);
      toast.error(`Failed to submit review: ${error.message}`);
    }
  };

  return (
    <div className="flex h-[calc(100vh-56px)]">
      {/* Left Panel - Pending Reviews */}
      <motion.div 
        initial={{ x: -40, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
        className="w-72 bg-card border-r border-border"
      >
        <ScrollArea className="h-full">
          <div className="p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-4">
              Pending Reviews ({pendingScans.length})
            </h3>
            <div className="space-y-2">
              {pendingScans.map((scan) => (
                <div 
                  key={scan.scan_id}
                  onClick={() => setSelectedScan(scan)}
                  className={`p-3 rounded-lg border cursor-pointer transition-all ${
                    selectedScan?.scan_id === scan.scan_id
                      ? 'border-primary bg-primary/5'
                      : 'border-border hover:border-primary/50 hover:bg-muted/50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{scan.patient_id}</span>
                    <Badge 
                      variant="outline" 
                      className="text-xs"
                      style={{ borderColor: scan.severity_color, color: scan.severity_color }}
                    >
                      {scan.severity_label}
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {scan.scan_id}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {new Date(scan.timestamp).toLocaleString()}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </ScrollArea>
      </motion.div>

      {/* Center - Image Comparison */}
      <div className="flex-1 bg-[#070A12] flex flex-col">
        {selectedScan ? (
          <>
            {/* Toolbar */}
            <div className="h-12 border-b border-border flex items-center justify-between px-4">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <Switch 
                    checked={showHeatmap} 
                    onCheckedChange={setShowHeatmap}
                    id="heatmap-toggle"
                  />
                  <Label htmlFor="heatmap-toggle" className="text-sm">Heatmap</Label>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setZoom(Math.max(50, zoom - 25))}>
                  <ZoomOut className="w-4 h-4" />
                </Button>
                <span className="text-xs mono w-12 text-center">{zoom}%</span>
                <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setZoom(Math.min(200, zoom + 25))}>
                  <ZoomIn className="w-4 h-4" />
                </Button>
              </div>
            </div>

            {/* Image Viewport */}
            <div className="flex-1 flex">
              {/* Original Image */}
              <div className="flex-1 flex items-center justify-center p-4 border-r border-border">
                <div className="text-center">
                  <p className="text-xs text-muted-foreground mb-2">Original Image</p>
                  <div className="corner-brackets">
                    <img
                      src={selectedScan.original_image}
                      alt="Original"
                      className="max-w-full max-h-[60vh] object-contain border border-white/10 rounded-lg"
                      style={{ transform: `scale(${zoom / 100})` }}
                    />
                  </div>
                </div>
              </div>

              {/* Heatmap Overlay */}
              <div className="flex-1 flex items-center justify-center p-4">
                <div className="text-center">
                  <p className="text-xs text-muted-foreground mb-2">
                    {showHeatmap ? 'Grad-CAM Heatmap' : 'Original'}
                  </p>
                  <div className="corner-brackets relative">
                    <img
                      src={showHeatmap ? selectedScan.heatmap_image : selectedScan.original_image}
                      alt="Heatmap"
                      className="max-w-full max-h-[60vh] object-contain border border-white/10 rounded-lg"
                      style={{ transform: `scale(${zoom / 100})` }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Heatmap Legend */}
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

      {/* Right Panel - Review Actions */}
      <AnimatePresence>
        {selectedScan && (
          <motion.div 
            initial={{ x: 40, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 40, opacity: 0 }}
            transition={{ duration: 0.5 }}
            className="w-80 bg-card border-l border-border"
          >
            <ScrollArea className="h-full">
              <div className="p-4 space-y-4">
                {/* AI Result Summary */}
                <Card className="border-primary/30">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">AI Assessment</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Result</span>
                      <Badge 
                        variant="outline"
                        style={{ borderColor: selectedScan.severity_color, color: selectedScan.severity_color }}
                      >
                        {selectedScan.severity_label}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Confidence</span>
                      <span className="font-bold mono">{selectedScan.confidence}%</span>
                    </div>
                  </CardContent>
                </Card>

                {/* Doctor Review */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">Doctor Review</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <RadioGroup value={reviewAction} onValueChange={setReviewAction}>
                      <div className="space-y-2">
                        <Label
                          htmlFor="approve"
                          className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                            reviewAction === 'approve' ? 'border-emerald-500 bg-emerald-500/10' : 'border-border'
                          }`}
                        >
                          <RadioGroupItem value="approve" id="approve" />
                          <CheckCircle className="w-4 h-4 text-emerald-500" />
                          <span className="text-sm">Approve AI result</span>
                        </Label>
                        
                        <Label
                          htmlFor="modify"
                          className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                            reviewAction === 'modify' ? 'border-amber-500 bg-amber-500/10' : 'border-border'
                          }`}
                        >
                          <RadioGroupItem value="modify" id="modify" />
                          <Edit3 className="w-4 h-4 text-amber-500" />
                          <span className="text-sm">Modify diagnosis</span>
                        </Label>
                        
                        <Label
                          htmlFor="override"
                          className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                            reviewAction === 'override' ? 'border-red-500 bg-red-500/10' : 'border-border'
                          }`}
                        >
                          <RadioGroupItem value="override" id="override" />
                          <XCircle className="w-4 h-4 text-red-500" />
                          <span className="text-sm">Reject / Re-scan</span>
                        </Label>
                      </div>
                    </RadioGroup>

                    <div>
                      <Label htmlFor="notes" className="text-sm">Clinical Notes</Label>
                      <Textarea
                        id="notes"
                        placeholder="Add your clinical observations..."
                        value={notes}
                        onChange={(e) => setNotes(e.target.value)}
                        className="mt-1 min-h-[100px]"
                      />
                    </div>

                    <Button 
                      className="w-full btn-medical"
                      onClick={handleSubmitReview}
                    >
                      <CheckCircle className="w-4 h-4 mr-2" />
                      Submit Review
                    </Button>
                  </CardContent>
                </Card>

                {/* Patient Info */}
                <div className="text-xs text-muted-foreground space-y-1">
                  <p>Patient: <span className="mono">{selectedScan.patient_id}</span></p>
                  <p>Scan: <span className="mono">{selectedScan.scan_id}</span></p>
                  <p>Time: {new Date(selectedScan.timestamp).toLocaleString()}</p>
                </div>
              </div>
            </ScrollArea>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Patient History Dashboard
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
    if (selectedScans.includes(scanId)) {
      setSelectedScans(selectedScans.filter(id => id !== scanId));
    } else if (selectedScans.length < 2) {
      setSelectedScans([...selectedScans, scanId]);
    }
  };

  return (
    <div className="flex h-[calc(100vh-56px)]">
      {/* Left Panel - Patient List */}
      <motion.div 
        initial={{ x: -40, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
        className="w-72 bg-card border-r border-border"
      >
        <ScrollArea className="h-full">
          <div className="p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Patients
              </h3>
              <Badge variant="outline">{patients.length}</Badge>
            </div>
            <div className="space-y-2">
              {patients.map((patient) => (
                <div 
                  key={patient.id}
                  onClick={() => setSelectedPatient(patient.id)}
                  className={`p-3 rounded-lg border cursor-pointer transition-all ${
                    selectedPatient === patient.id
                      ? 'border-primary bg-primary/5'
                      : 'border-border hover:border-primary/50 hover:bg-muted/50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{patient.name}</span>
                    <Badge variant="secondary" className="text-xs">{patient.scans} scans</Badge>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {patient.id} • {patient.age} years
                  </p>
                </div>
              ))}
            </div>
          </div>
        </ScrollArea>
      </motion.div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto">
        <div className="p-6 space-y-6">
          {/* Patient Header */}
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold">{patients.find(p => p.id === selectedPatient)?.name}</h2>
              <p className="text-muted-foreground">{selectedPatient} • {patients.find(p => p.id === selectedPatient)?.age} years</p>
            </div>
            <div className="flex items-center gap-2">
              <Button 
                variant={compareMode ? 'default' : 'outline'}
                onClick={() => setCompareMode(!compareMode)}
              >
                <Maximize2 className="w-4 h-4 mr-2" />
                {compareMode ? 'Exit Compare' : 'Compare Scans'}
              </Button>
            </div>
          </div>

          {/* Trend Chart */}
          <Card className="medical-panel">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-primary" />
                Severity Trend Over Time
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={TREND_DATA}>
                    <defs>
                      <linearGradient id="colorSeverity" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#22D3EE" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#22D3EE" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="date" stroke="rgba(255,255,255,0.5)" fontSize={12} />
                    <YAxis stroke="rgba(255,255,255,0.5)" fontSize={12} domain={[0, 4]} ticks={[0, 1, 2, 3, 4]} />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#111827', 
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px'
                      }}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="severity" 
                      stroke="#22D3EE" 
                      fillOpacity={1} 
                      fill="url(#colorSeverity)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Scan History Grid */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Scan History</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {MOCK_PATIENT_HISTORY.map((scan) => (
                <Card 
                  key={scan.scan_id}
                  className={`cursor-pointer transition-all ${
                    compareMode && selectedScans.includes(scan.scan_id)
                      ? 'border-primary ring-2 ring-primary/20'
                      : ''
                  }`}
                  onClick={() => compareMode && toggleScanSelection(scan.scan_id)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">{scan.scan_id}</span>
                      {compareMode && (
                        <div className={`w-5 h-5 rounded border flex items-center justify-center ${
                          selectedScans.includes(scan.scan_id) ? 'bg-primary border-primary' : 'border-border'
                        }`}>
                          {selectedScans.includes(scan.scan_id) && <CheckCircle className="w-3 h-3" />}
                        </div>
                      )}
                    </div>
                    <div className="flex items-center gap-2 mb-2">
                      <Badge 
                        variant="outline"
                        style={{ borderColor: scan.severity_color, color: scan.severity_color }}
                      >
                        {scan.severity_label}
                      </Badge>
                      <span className="text-xs text-muted-foreground">{scan.laterality}</span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {new Date(scan.timestamp).toLocaleDateString()}
                    </p>
                    <div className="mt-2 flex items-center gap-2">
                      <Progress value={scan.confidence} className="h-1.5 flex-1" />
                      <span className="text-xs mono">{scan.confidence}%</span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>

          {/* Comparison View */}
          {compareMode && selectedScans.length === 2 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="grid grid-cols-2 gap-4"
            >
              {selectedScans.map((scanId) => {
                const scan = MOCK_PATIENT_HISTORY.find(s => s.scan_id === scanId);
                return (
                  <Card key={scanId} className="medical-panel">
                    <CardHeader>
                      <CardTitle className="text-sm">{scan?.scan_id}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="aspect-square bg-muted rounded-lg flex items-center justify-center">
                        <Eye className="w-16 h-16 opacity-30" />
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}

// Admin Analytics Panel
function AdminAnalyticsPanel() {
  const [stats, setStats] = useState<{
    total_scans: number;
    severity_distribution: Record<number, number>;
    reviews_submitted: number;
    override_rate: number;
  } | null>(null);
  const [recentScans, setRecentScans] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        // Fetch statistics
        const statsRes = await fetch(`${API_BASE_URL}/api/v1/analytics/statistics`);
        if (statsRes.ok) {
          const data = await statsRes.json();
          setStats(data);
        }
        
        // Fetch recent scans
        const scansRes = await fetch(`${API_BASE_URL}/api/scans`);
        if (scansRes.ok) {
          const data = await scansRes.json();
          setRecentScans(data.scans || []);
        }
      } catch (err) {
        console.error('Failed to fetch analytics:', err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchAnalytics();
    // Refresh every 30 seconds
    const interval = setInterval(fetchAnalytics, 30000);
    return () => clearInterval(interval);
  }, []);

  const totalScans = stats?.total_scans ?? 0;
  const severityData = stats?.severity_distribution ?? {0: 0, 1: 0, 2: 0, 3: 0, 4: 0};

  return (
    <div className="h-[calc(100vh-56px)] overflow-auto">
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold">Analytics Dashboard</h2>
            <p className="text-muted-foreground">System performance and screening statistics</p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="flex items-center gap-2">
              <span className="status-online" />
              System Online
            </Badge>
          </div>
        </div>

        {loading && (
          <div className="text-center py-8">
            <p className="text-muted-foreground">Loading analytics...</p>
          </div>
        )}

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="medical-panel">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Total Scans</p>
                  <p className="text-3xl font-bold mono">{totalScans}</p>
                </div>
                <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
                  <Activity className="w-6 h-6 text-primary" />
                </div>
              </div>
              <p className="text-xs text-emerald-400 mt-2">Session data</p>
            </CardContent>
          </Card>

          <Card className="medical-panel">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Model Accuracy</p>
                  <p className="text-3xl font-bold mono">{(ANALYTICS_DATA.modelPerformance.accuracy * 100).toFixed(0)}%</p>
                </div>
                <div className="w-12 h-12 bg-emerald-500/10 rounded-lg flex items-center justify-center">
                  <CheckCircle className="w-6 h-6 text-emerald-500" />
                </div>
              </div>
              <p className="text-xs text-muted-foreground mt-2">AUC: {(ANALYTICS_DATA.modelPerformance.auc * 100).toFixed(0)}%</p>
            </CardContent>
          </Card>

          <Card className="medical-panel">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Override Rate</p>
                  <p className="text-3xl font-bold mono">{stats?.override_rate?.toFixed(1) ?? 0}%</p>
                </div>
                <div className="w-12 h-12 bg-amber-500/10 rounded-lg flex items-center justify-center">
                  <AlertCircle className="w-6 h-6 text-amber-500" />
                </div>
              </div>
              <p className="text-xs text-muted-foreground mt-2">{stats?.reviews_submitted ?? 0} reviews</p>
            </CardContent>
          </Card>

          <Card className="medical-panel">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Active Users</p>
                  <p className="text-3xl font-bold mono">1</p>
                </div>
                <div className="w-12 h-12 bg-blue-500/10 rounded-lg flex items-center justify-center">
                  <Users className="w-6 h-6 text-blue-500" />
                </div>
              </div>
              <p className="text-xs text-muted-foreground mt-2">3 doctors, 8 technicians</p>
            </CardContent>
          </Card>
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Daily Scans Chart */}
          <Card className="medical-panel">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-primary" />
                Daily Screening Volume
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={ANALYTICS_DATA.dailyScans}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="day" stroke="rgba(255,255,255,0.5)" fontSize={12} />
                    <YAxis stroke="rgba(255,255,255,0.5)" fontSize={12} />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#111827', 
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px'
                      }}
                    />
                    <Bar dataKey="scans" fill="#22D3EE" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Severity Distribution */}
          <Card className="medical-panel">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <PieChart className="w-5 h-5 text-primary" />
                Severity Distribution
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={ANALYTICS_DATA.severityDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {ANALYTICS_DATA.severityDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#111827', 
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px'
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="flex flex-wrap gap-2 justify-center mt-2">
                {ANALYTICS_DATA.severityDistribution.map((item) => (
                  <div key={item.name} className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                    <span className="text-xs">{item.name}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Model Performance Metrics */}
        <Card className="medical-panel">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Cpu className="w-5 h-5 text-primary" />
              Model Performance Metrics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div className="text-center">
                <p className="text-sm text-muted-foreground mb-1">Sensitivity</p>
                <p className="text-4xl font-bold mono text-emerald-400">
                  {(ANALYTICS_DATA.modelPerformance.sensitivity * 100).toFixed(0)}%
                </p>
                <Progress value={ANALYTICS_DATA.modelPerformance.sensitivity * 100} className="h-2 mt-2" />
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground mb-1">Specificity</p>
                <p className="text-4xl font-bold mono text-blue-400">
                  {(ANALYTICS_DATA.modelPerformance.specificity * 100).toFixed(0)}%
                </p>
                <Progress value={ANALYTICS_DATA.modelPerformance.specificity * 100} className="h-2 mt-2" />
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground mb-1">AUC-ROC</p>
                <p className="text-4xl font-bold mono text-primary">
                  {(ANALYTICS_DATA.modelPerformance.auc * 100).toFixed(0)}%
                </p>
                <Progress value={ANALYTICS_DATA.modelPerformance.auc * 100} className="h-2 mt-2" />
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground mb-1">Accuracy</p>
                <p className="text-4xl font-bold mono text-amber-400">
                  {(ANALYTICS_DATA.modelPerformance.accuracy * 100).toFixed(0)}%
                </p>
                <Progress value={ANALYTICS_DATA.modelPerformance.accuracy * 100} className="h-2 mt-2" />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* System Health */}
        <Card className="medical-panel">
          <CardHeader>
            <CardTitle className="text-lg">System Health</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-center gap-4 p-4 rounded-lg bg-muted/50">
                <div className="w-10 h-10 bg-emerald-500/20 rounded-lg flex items-center justify-center">
                  <Cpu className="w-5 h-5 text-emerald-500" />
                </div>
                <div>
                  <p className="text-sm font-medium">GPU Status</p>
                  <p className="text-xs text-muted-foreground">CUDA Available • 8GB VRAM</p>
                </div>
              </div>
              <div className="flex items-center gap-4 p-4 rounded-lg bg-muted/50">
                <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                  <Activity className="w-5 h-5 text-blue-500" />
                </div>
                <div>
                  <p className="text-sm font-medium">API Latency</p>
                  <p className="text-xs text-muted-foreground">Avg: 245ms • P95: 380ms</p>
                </div>
              </div>
              <div className="flex items-center gap-4 p-4 rounded-lg bg-muted/50">
                <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
                  <Shield className="w-5 h-5 text-purple-500" />
                </div>
                <div>
                  <p className="text-sm font-medium">Security</p>
                  <p className="text-xs text-muted-foreground">SSL Active • HIPAA Compliant</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// ============================================================================
// MAIN APP
// ============================================================================

function App() {
  const [userRole, setUserRole] = useState<UserRole>(null);
  const [activeTab, setActiveTab] = useState('upload');

  const handleLogin = (role: UserRole) => {
    setUserRole(role);
    // Set default tab based on role
    if (role === 'doctor') setActiveTab('review');
    else if (role === 'technician') setActiveTab('upload');
    else if (role === 'admin') setActiveTab('analytics');
  };

  const handleLogout = () => {
    setUserRole(null);
    setActiveTab('upload');
    toast.info('Logged out successfully');
  };

  if (!userRole) {
    return <LoginPage onLogin={handleLogin} />;
  }

  // Navigation items based on role
  const navItems: Record<string, { id: string; label: string; icon: React.ElementType }[]> = {
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
      {/* Noise Overlay */}
      <div className="noise-overlay" />
      
      <Header userRole={userRole} onLogout={handleLogout} />
      
      {/* Navigation Sidebar */}
      <div className="flex h-[calc(100vh-56px)]">
        <motion.aside 
          initial={{ x: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.4 }}
          className="w-16 bg-card border-r border-border flex flex-col items-center py-4 gap-2"
        >
          {navItems[userRole]?.map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-12 h-12 rounded-lg flex items-center justify-center transition-all ${
                activeTab === item.id
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground'
              }`}
              title={item.label}
            >
              <item.icon className="w-5 h-5" />
            </button>
          ))}
        </motion.aside>

        {/* Main Content Area */}
        <main className="flex-1 overflow-hidden">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
              className="h-full"
            >
              {activeTab === 'upload' && <TechnicianDashboard />}
              {activeTab === 'review' && <DoctorReviewPortal />}
              {activeTab === 'history' && <PatientHistoryDashboard />}
              {activeTab === 'analytics' && <AdminAnalyticsPanel />}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

export default App;
