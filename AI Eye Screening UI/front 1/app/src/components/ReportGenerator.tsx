import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';
import {
  FileText,
  Download,
  Printer,
  CheckCircle,
  AlertCircle,
  X,
  Stethoscope,
  Eye,
  Activity,
  TrendingUp,
  Shield,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';

interface ReportData {
  scan_id: string;
  patient_id: string;
  patient_name: string;
  patient_age: number;
  patient_gender: string;
  date: string;
  laterality: string;
  binary_result: string;
  confidence: number;
  severity_level: number;
  severity_label: string;
  severity_color: string;
  multi_disease: Record<string, { detected: boolean; confidence: number }>;
  doctor_notes?: string;
  doctor_name?: string;
  original_image?: string;
  heatmap_image?: string;
}

// Default mock data
const DEFAULT_REPORT: ReportData = {
  scan_id: 'SCAN-A7B2C9D1',
  patient_id: 'P-2026-0041',
  patient_name: 'John Anderson',
  patient_age: 58,
  patient_gender: 'Male',
  date: '2026-04-18',
  laterality: 'OD',
  binary_result: 'DR Detected',
  confidence: 91.5,
  severity_level: 2,
  severity_label: 'Moderate',
  severity_color: '#F97316',
  multi_disease: {
    amd: { detected: false, confidence: 0.12 },
    glaucoma: { detected: false, confidence: 0.28 },
    cataract: { detected: true, confidence: 0.41 },
  },
  doctor_notes: '',
  doctor_name: 'Dr. Sarah Mitchell',
  original_image: '',
  heatmap_image: '',
};

// Severity recommendations
const SEVERITY_RECOMMENDATIONS: Record<number, string> = {
  0: 'No diabetic retinopathy detected. Continue routine annual screening.',
  1: 'Mild NPDR detected. Follow-up examination recommended in 6-12 months.',
  2: 'Moderate NPDR detected. Ophthalmology referral recommended within 3 months. Monitor closely for progression.',
  3: 'Severe NPDR detected. Urgent ophthalmology referral within 2-4 weeks. Pan-retinal photocoagulation may be indicated.',
  4: 'Proliferative DR detected. Immediate referral to retinal specialist. Treatment required to prevent vision loss.',
};

interface ReportGeneratorProps {
  data?: ReportData;
  onClose?: () => void;
}

export default function ReportGenerator({ data = DEFAULT_REPORT, onClose }: ReportGeneratorProps) {
  const [doctorNotes, setDoctorNotes] = useState(data.doctor_notes || '');
  const [doctorName, setDoctorName] = useState(data.doctor_name || '');
  const [isGenerating, setIsGenerating] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const reportRef = useRef<HTMLDivElement>(null);

  const handleGeneratePDF = async () => {
    if (!reportRef.current) return;
    setIsGenerating(true);

    try {
      const canvas = await html2canvas(reportRef.current, {
        scale: 2,
        backgroundColor: '#ffffff',
        logging: false,
      });

      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = pdf.internal.pageSize.getHeight();
      const imgWidth = canvas.width;
      const imgHeight = canvas.height;
      const ratio = Math.min(pdfWidth / imgWidth, (pdfHeight - 20) / imgHeight);

      pdf.addImage(
        imgData,
        'PNG',
        (pdfWidth - imgWidth * ratio) / 2,
        10,
        imgWidth * ratio,
        imgHeight * ratio
      );

      pdf.save(`EYE-ASSISST_Report_${data.patient_id}_${data.date}.pdf`);
      toast('Report downloaded successfully');
    } catch (error) {
      toast('Failed to generate PDF');
    } finally {
      setIsGenerating(false);
    }
  };

  const handlePrint = () => {
    window.print();
  };

  const severityColors: Record<number, { bg: string; text: string; border: string }> = {
    0: { bg: 'bg-emerald-500/10', text: 'text-emerald-500', border: 'border-emerald-500/30' },
    1: { bg: 'bg-amber-500/10', text: 'text-amber-500', border: 'border-amber-500/30' },
    2: { bg: 'bg-orange-500/10', text: 'text-orange-500', border: 'border-orange-500/30' },
    3: { bg: 'bg-red-500/10', text: 'text-red-500', border: 'border-red-500/30' },
    4: { bg: 'bg-red-600/10', text: 'text-red-600', border: 'border-red-600/30' },
  };

  const colors = severityColors[data.severity_level] || severityColors[0];

  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* Actions Bar */}
      <div className="flex items-center justify-between mb-6 print:hidden">
        <div className="flex items-center gap-2">
          <FileText className="w-5 h-5 text-primary" />
          <h2 className="text-lg font-semibold">Clinical Report</h2>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowPreview(!showPreview)}
          >
            {showPreview ? 'Edit' : 'Preview'}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handlePrint}
          >
            <Printer className="w-4 h-4 mr-1" />
            Print
          </Button>
          <Button
            size="sm"
            onClick={handleGeneratePDF}
            disabled={isGenerating}
            className="btn-medical"
          >
            {isGenerating ? (
              <>
                <Activity className="w-4 h-4 mr-1 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Download className="w-4 h-4 mr-1" />
                Download PDF
              </>
            )}
          </Button>
          {onClose && (
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="w-4 h-4" />
            </Button>
          )}
        </div>
      </div>

      {/* Doctor Input Section (hidden in preview) */}
      <AnimatePresence>
        {!showPreview && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-6 space-y-4 print:hidden overflow-hidden"
          >
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Stethoscope className="w-4 h-4 text-primary" />
                  Doctor Review
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm text-muted-foreground mb-1 block">Doctor Name</label>
                  <input
                    type="text"
                    value={doctorName}
                    onChange={(e) => setDoctorName(e.target.value)}
                    placeholder="Enter your name"
                    className="w-full px-3 py-2 bg-background border border-border rounded-md text-sm"
                  />
                </div>
                <div>
                  <label className="text-sm text-muted-foreground mb-1 block">Clinical Notes & Recommendations</label>
                  <Textarea
                    value={doctorNotes}
                    onChange={(e) => setDoctorNotes(e.target.value)}
                    placeholder="Add your clinical observations, recommendations, and follow-up plans..."
                    className="min-h-[100px]"
                  />
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Report Preview */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div
          ref={reportRef}
          className="bg-white text-gray-900 p-8 rounded-lg shadow-lg"
          style={{ minHeight: '800px' }}
        >
          {/* Header */}
          <div className="flex items-start justify-between mb-8 border-b-2 border-gray-200 pb-6">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center">
                <Eye className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">EYE-ASSISST</h1>
                <p className="text-sm text-gray-500">AI-Powered Retinal Disease Screening Report</p>
              </div>
            </div>
            <div className="text-right">
              <Badge variant="outline" className="mb-2 text-xs">
                <Shield className="w-3 h-3 mr-1" />
                HIPAA Compliant
              </Badge>
              <p className="text-xs text-gray-500">Confidential Medical Document</p>
            </div>
          </div>

          {/* Patient Info */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8 p-4 bg-gray-50 rounded-lg">
            <div>
              <p className="text-xs text-gray-500 uppercase">Patient ID</p>
              <p className="font-semibold">{data.patient_id}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500 uppercase">Name</p>
              <p className="font-semibold">{data.patient_name}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500 uppercase">Age / Gender</p>
              <p className="font-semibold">{data.patient_age} / {data.patient_gender}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500 uppercase">Date</p>
              <p className="font-semibold">{data.date}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500 uppercase">Scan ID</p>
              <p className="font-semibold text-xs">{data.scan_id}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500 uppercase">Eye</p>
              <p className="font-semibold">{data.laterality === 'OD' ? 'Right Eye (OD)' : 'Left Eye (OS)'}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500 uppercase">Reviewed By</p>
              <p className="font-semibold">{doctorName || 'Pending Review'}</p>
            </div>
          </div>

          {/* AI Findings */}
          <div className="mb-8">
            <h2 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-cyan-600" />
              AI Findings
            </h2>

            {/* Primary Result */}
            <div className={`p-4 rounded-lg border-2 mb-4 ${colors.bg} ${colors.border}`}>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Primary Diagnosis</p>
                  <p className={`text-2xl font-bold ${colors.text}`}>
                    {data.severity_label} Diabetic Retinopathy
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-600">Confidence</p>
                  <p className="text-3xl font-bold text-gray-900">{data.confidence}%</p>
                </div>
              </div>
            </div>

            {/* Multi-disease Screening */}
            <div className="grid grid-cols-3 gap-4">
              {Object.entries(data.multi_disease).map(([disease, result]) => (
                <div
                  key={disease}
                  className={`p-4 rounded-lg border ${
                    result.detected
                      ? 'bg-amber-50 border-amber-300'
                      : 'bg-gray-50 border-gray-200'
                  }`}
                >
                  <p className="text-xs text-gray-500 uppercase">{disease.toUpperCase()}</p>
                  <div className="flex items-center gap-2 mt-1">
                    {result.detected ? (
                      <AlertCircle className="w-4 h-4 text-amber-600" />
                    ) : (
                      <CheckCircle className="w-4 h-4 text-emerald-600" />
                    )}
                    <span className={`font-semibold ${result.detected ? 'text-amber-700' : 'text-emerald-700'}`}>
                      {result.detected ? 'Detected' : 'Negative'}
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Confidence: {(result.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Images */}
          <div className="mb-8">
            <h2 className="text-lg font-bold text-gray-900 mb-4">Imaging Analysis</h2>
            <div className="grid grid-cols-2 gap-6">
              <div>
                <p className="text-sm text-gray-600 mb-2 text-center">Original Fundus Image</p>
                <div className="aspect-square bg-gray-100 rounded-lg flex items-center justify-center border border-gray-200">
                  {data.original_image ? (
                    <img src={data.original_image} alt="Fundus" className="w-full h-full object-contain rounded-lg" />
                  ) : (
                    <div className="text-center text-gray-400">
                      <Eye className="w-12 h-12 mx-auto mb-2" />
                      <p className="text-sm">Fundus Image</p>
                    </div>
                  )}
                </div>
              </div>
              <div>
                <p className="text-sm text-gray-600 mb-2 text-center">Grad-CAM Heatmap</p>
                <div className="aspect-square bg-gray-100 rounded-lg flex items-center justify-center border border-gray-200">
                  {data.heatmap_image ? (
                    <img src={data.heatmap_image} alt="Heatmap" className="w-full h-full object-contain rounded-lg" />
                  ) : (
                    <div className="text-center text-gray-400">
                      <TrendingUp className="w-12 h-12 mx-auto mb-2" />
                      <p className="text-sm">AI Attention Map</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Recommendation */}
          <div className="mb-8">
            <h2 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-amber-600" />
              Clinical Recommendation
            </h2>
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-gray-800 leading-relaxed">
                {SEVERITY_RECOMMENDATIONS[data.severity_level] || SEVERITY_RECOMMENDATIONS[0]}
              </p>
            </div>
          </div>

          {/* Doctor Notes */}
          {(doctorNotes || showPreview) && (
            <div className="mb-8">
              <h2 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                <Stethoscope className="w-5 h-5 text-gray-600" />
                Physician Notes
              </h2>
              <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                <p className="text-gray-800 whitespace-pre-wrap">
                  {doctorNotes || 'No additional notes provided.'}
                </p>
              </div>
            </div>
          )}

          {/* Footer */}
          <div className="mt-12 pt-6 border-t border-gray-200">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center">
                  <Eye className="w-4 h-4 text-white" />
                </div>
                <div>
                  <p className="text-sm font-semibold">EYE-ASSISST</p>
                  <p className="text-xs text-gray-500">AI-Powered Eye Disease Screening</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-xs text-gray-500">Report generated by AI system</p>
                <p className="text-xs text-gray-500">Requires physician verification</p>
              </div>
            </div>
            <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded text-xs text-amber-800">
              <strong>Disclaimer:</strong> This report was generated using artificial intelligence and is intended to assist
              clinical decision-making. It does not replace professional medical judgment. Please verify all findings
              and use your clinical expertise when making diagnostic and treatment decisions.
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

// Quick inline toast function
function toast(message: string) {
  // Simple toast implementation
  const div = document.createElement('div');
  div.className = 'fixed bottom-4 right-4 bg-gray-900 text-white px-4 py-2 rounded-lg shadow-lg z-50 text-sm';
  div.textContent = message;
  document.body.appendChild(div);
  setTimeout(() => div.remove(), 3000);
}
