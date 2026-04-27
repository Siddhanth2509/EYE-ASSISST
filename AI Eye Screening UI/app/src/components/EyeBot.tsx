import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MessageCircle,
  X,
  Send,
  Bot,
  User,
  Sparkles,
  ChevronRight,
  Trash2,
} from 'lucide-react';
import { Button } from '@/components/ui/button';

// Knowledge Base
interface KnowledgeEntry {
  keywords: string[];
  response: string;
  quickReplies?: string[];
}

const KNOWLEDGE_BASE: KnowledgeEntry[] = [
  {
    keywords: ['hello', 'hi', 'hey', 'greetings'],
    response: "Hello! I'm EyeBot, your AI assistant for eye disease screening. I can help you understand retinal conditions, explain analysis results, or guide you through the screening process. How can I help today?",
    quickReplies: ['What is DR?', 'How does screening work?', 'Book appointment'],
  },
  {
    keywords: ['dr', 'diabetic retinopathy', 'diabetic'],
    response: "Diabetic Retinopathy (DR) is a diabetes complication that affects the eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina).\n\n**Stages:**\n• **Normal** - No signs of DR\n• **Mild NPDR** - Microaneurysms only\n• **Moderate NPDR** - More severe microaneurysms, hemorrhages\n• **Severe NPDR** - Any of the following: >20 hemorrhages, venous beading, IRMA\n• **Proliferative DR** - Neovascularization, vitreous hemorrhage\n\nEarly detection through regular screening is crucial to prevent vision loss.",
    quickReplies: ['What is AMD?', 'How is DR treated?', 'Screening process'],
  },
  {
    keywords: ['amd', 'macular degeneration', 'age-related'],
    response: "Age-related Macular Degeneration (AMD) is a common eye condition and a leading cause of vision loss among people age 50 and older. It causes damage to the macula, a small spot near the center of the retina.\n\n**Types:**\n• **Dry AMD** (80% of cases) - Gradual thinning of macula tissue\n• **Wet AMD** (10-15%) - Abnormal blood vessel growth under retina\n\n**Risk Factors:**\n• Age over 50\n• Family history\n• Smoking\n• High blood pressure\n• Obesity\n\nRegular eye exams are essential for early detection.",
    quickReplies: ['DR vs AMD', 'AMD treatment', 'Risk factors'],
  },
  {
    keywords: ['glaucoma', 'optic nerve', 'intraocular pressure'],
    response: "Glaucoma is a group of eye conditions that damage the optic nerve, often caused by abnormally high pressure in your eye. It's one of the leading causes of blindness for people over 60.\n\n**Types:**\n• **Open-angle glaucoma** - Gradual, often no symptoms until advanced\n• **Angle-closure glaucoma** - Sudden, medical emergency\n• **Normal-tension glaucoma** - Optic nerve damage despite normal pressure\n\n**Warning Signs:**\n• Patchy blind spots in peripheral vision\n• Tunnel vision in advanced stages\n• Severe headache and eye pain (acute angle-closure)\n\nRegular screening is vital as early glaucoma has no symptoms.",
    quickReplies: ['Glaucoma treatment', 'Eye pressure', 'Screening'],
  },
  {
    keywords: ['cataract', 'cloudy lens', 'lens opacity'],
    response: "A cataract is a clouding of the normally clear lens of your eye. It's like looking through a frosty or fogged-up window. Cataracts commonly develop slowly and don't disturb eyesight early on.\n\n**Symptoms:**\n• Clouded, blurred or dim vision\n• Increasing difficulty with vision at night\n• Sensitivity to light and glare\n• Fading or yellowing of colors\n• Double vision in a single eye\n\n**Treatment:** Cataract surgery is a safe, effective procedure where the cloudy lens is removed and replaced with a clear artificial lens.",
    quickReplies: ['Cataract surgery', 'Prevention', 'Other diseases'],
  },
  {
    keywords: ['screening', 'process', 'how does it work', 'workflow'],
    response: "Our AI screening process is simple and takes under 60 seconds:\n\n**Step 1: Upload**\nDrag and drop a fundus photograph into the upload area. We support JPEG, PNG, and DICOM formats.\n\n**Step 2: AI Analysis**\nOur deep learning model analyzes the image, detecting signs of 6 major eye diseases. The AI highlights areas of concern using Grad-CAM heatmaps.\n\n**Step 3: Review**\nA doctor reviews the AI findings, confirms or modifies the diagnosis, and adds clinical notes.\n\n**Step 4: Report**\nA comprehensive PDF report is generated for the patient record.\n\nThe entire process is HIPAA-compliant and secure.",
    quickReplies: ['What diseases?', 'Upload help', 'Report format'],
  },
  {
    keywords: ['grad-cam', 'heatmap', 'attention map', 'explanation'],
    response: "Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique that shows which parts of the image the AI focused on when making its decision.\n\n**How to read the heatmap:**\n• **Red/Orange areas** - High AI attention, potential pathology\n• **Yellow areas** - Moderate attention\n• **Transparent areas** - Low attention, likely normal tissue\n\nThe heatmap helps doctors understand *why* the AI made a particular diagnosis, making the AI decision transparent and interpretable. This builds trust and helps catch any potential AI errors.",
    quickReplies: ['AI accuracy', 'Trust AI', 'Heatmap colors'],
  },
  {
    keywords: ['accuracy', 'performance', 'auc', 'sensitivity', 'specificity'],
    response: "EYE-ASSISST has been validated on large clinical datasets and achieves state-of-the-art performance:\n\n**Overall Metrics:**\n• **AUC-ROC:** 96.4%\n• **Sensitivity:** 94.2% (catches 94% of true cases)\n• **Specificity:** 89.3% (correctly identifies 89% of normal cases)\n• **Accuracy:** 91.5%\n\n**Per-Disease AUC:**\n• Diabetic Retinopathy: 97.1%\n• AMD: 94.8%\n• Glaucoma: 93.5%\n• Cataract: 92.7%\n\nHowever, AI is a decision support tool - all findings should be verified by a qualified ophthalmologist.",
    quickReplies: ['Validation study', 'False positives', 'Compare doctors'],
  },
  {
    keywords: ['treatment', 'treated', 'therapy', 'management'],
    response: "Treatment depends on the specific condition and severity:\n\n**Diabetic Retinopathy:**\n• Mild-Moderate: Tight glucose control, regular monitoring\n• Severe: Anti-VEGF injections, laser photocoagulation\n• Proliferative: Pan-retinal photocoagulation, vitrectomy\n\n**AMD:**\n• Dry AMD: AREDS2 supplements, lifestyle changes\n• Wet AMD: Anti-VEGF injections (monthly)\n\n**Glaucoma:**\n• Eye drops to lower pressure\n• Laser trabeculoplasty\n• Surgery (trabeculectomy, shunts)\n\n**Cataract:**\n• Surgery is the only effective treatment\n\nAlways consult with your ophthalmologist for personalized treatment plans.",
    quickReplies: ['Prevention', 'Lifestyle changes', 'Find doctor'],
  },
  {
    keywords: ['severity', 'grade', 'level', 'mild', 'moderate', 'severe'],
    response: "Disease severity grading helps guide treatment decisions:\n\n**DR Severity Scale (0-4):**\n• **Grade 0 (Normal)** - No DR signs. Annual screening.\n• **Grade 1 (Mild)** - Microaneurysms only. Screen every 6-12 months.\n• **Grade 2 (Moderate)** - More hemorrhages, hard exudates. Refer within 3 months.\n• **Grade 3 (Severe)** - >20 hemorrhages, venous beading. Urgent referral (2-4 weeks).\n• **Grade 4 (Proliferative)** - Neovascularization. Immediate specialist referral.\n\nHigher grades indicate more advanced disease requiring more urgent intervention.",
    quickReplies: ['My results', 'Treatment by grade', 'Progression'],
  },
  {
    keywords: ['appointment', 'book', 'schedule', 'consultation', 'doctor'],
    response: "To book a consultation with one of our specialists:\n\n**Available Time Slots (Tomorrow):**\n• 9:00 AM - Dr. Sarah Mitchell (Retina)\n• 11:30 AM - Dr. James Chen (Glaucoma)\n• 2:00 PM - Dr. Maria Garcia (General Ophthalmology)\n• 4:30 PM - Dr. Robert Kim (Cornea)\n\nYou can book through the Patient Portal or call our scheduling line. Urgent cases are typically seen within 24-48 hours.\n\nWould you like me to help you navigate to the booking page?",
    quickReplies: ['Patient portal', 'Emergency', 'Specialists'],
  },
  {
    keywords: ['prevention', 'prevent', 'lifestyle', 'diet', 'exercise'],
    response: "Lifestyle factors play a significant role in eye health:\n\n**For Diabetics (DR prevention):**\n• Maintain HbA1c < 7%\n• Control blood pressure (<130/80)\n• Regular cholesterol monitoring\n• Annual dilated eye exams\n\n**For AMD prevention:**\n• Eat leafy greens, fish, colorful fruits\n• Don't smoke (2-5x increased risk)\n• Maintain healthy weight\n• Wear UV-protective sunglasses\n\n**For Glaucoma:**\n• Regular eye pressure checks after 40\n• Family history awareness\n• Exercise regularly (lowers eye pressure)\n\n**General:**\n• 20-20-20 rule for screen use\n• Adequate sleep\n• Stay hydrated",
    quickReplies: ['Supplements', 'Screen time', 'Smoking effects'],
  },
  {
    keywords: ['confidence', 'score', 'percentage', '%'],
    response: "The confidence score indicates how certain the AI is about its prediction:\n\n**Interpreting Confidence:**\n• **90-100%** - High confidence, reliable prediction\n• **70-89%** - Moderate confidence, recommend doctor review\n• **50-69%** - Low confidence, image quality may be suboptimal\n• **Below 50%** - Uncertain, consider re-scanning\n\n**Important:** Confidence reflects the AI's certainty, not disease severity. A low-confidence severe diagnosis still requires urgent attention.\n\nFactors affecting confidence:\n• Image quality and focus\n• Disease pattern clarity\n• Similarity to training data\n• Presence of multiple conditions",
    quickReplies: ['Low confidence', 'Image quality', 'Re-scan'],
  },
  {
    keywords: ['help', 'support', 'contact', 'assistance'],
    response: "I can help you with:\n\n📋 **Understanding Diseases** - DR, AMD, Glaucoma, Cataract, and more\n📊 **Interpreting Results** - Severity grades, confidence scores\n🔍 **Screening Process** - How to upload, what to expect\n📅 **Appointments** - Booking with specialists\n💡 **Prevention Tips** - Lifestyle recommendations\n📖 **General Eye Health** - Common questions answered\n\nWhat would you like to know about?",
    quickReplies: ['What is DR?', 'Screening process', 'Book appointment'],
  },
  {
    keywords: ['referral', 'refer', 'when to refer', 'urgent'],
    response: "👨‍⚕️ **Clinical Referral Guidelines:**\n\n**Immediate/Urgent (Same day or < 24h):**\n• Sudden vision loss\n• Suspected acute angle-closure glaucoma\n• Retinal detachment (flashes/floaters + field loss)\n• Suspected wet AMD with new hemorrhage\n\n**Semi-Urgent (1-2 weeks):**\n• Severe NPDR or PDR (Grade 3-4)\n• Unexplained drop in visual acuity\n• Macular edema involving fovea\n\n**Routine (< 3 months):**\n• Moderate NPDR (Grade 2)\n• Glaucoma suspect (cupping without vision loss)\n• Progression of dry AMD\n• Visually significant cataract",
    quickReplies: ['Grade 3 protocol', 'Heatmap reading', 'Accuracy'],
  },
  {
    keywords: ['protocol', 'guideline', 'management plan'],
    response: "👨‍⚕️ **Standard Management Protocols by Severity:**\n\n**Grade 0 (Normal):** Annual rescreening.\n**Grade 1 (Mild NPDR):** Screen every 6-12 months. Strict glycemic control.\n**Grade 2 (Moderate NPDR):** Screen every 3-6 months. Consider OCT to rule out DME.\n**Grade 3 (Severe NPDR):** Urgent referral. High risk of progression. Panretinal photocoagulation (PRP) may be indicated.\n**Grade 4 (Proliferative):** Immediate referral. PRP and/or Anti-VEGF therapy required. Risk of tractional retinal detachment.",
    quickReplies: ['Referral guidelines', 'Interpret heatmap', 'False positives'],
  },
  {
    keywords: ['false positive', 'artifacts', 'image quality', 'suboptimal'],
    response: "👨‍⚕️ **Handling Suboptimal Images & Artifacts:**\n\n**Common Causes of False Positives:**\n• Dust on camera lens mimicking microaneurysms\n• Uneven illumination causing pseudo-exudates\n• Severe cataracts obscuring retinal details\n• Prominent choroidal vessels in lightly pigmented fundi\n\n**Recommendation:** If the AI confidence is <70% or the heatmap highlights obvious artifacts (like the image border or dust spots), please override the AI and request a rescan or refer for clinical examination.",
    quickReplies: ['Interpret heatmap', 'Confidence score', 'Referral guidelines'],
  }
];

// Default response when no match found
const DEFAULT_RESPONSE: KnowledgeEntry = {
  keywords: [],
  response: "I'm not sure I understood that correctly. I can help you with:\n\n• Understanding eye diseases (DR, AMD, Glaucoma, Cataract)\n• Explaining screening results and severity grades\n• Guiding you through the upload process\n• Booking appointments with specialists\n• General eye health and prevention tips\n\nCould you rephrase your question, or select one of the suggested topics below?",
  quickReplies: ['What is DR?', 'How does screening work?', 'Help'],
};

// Message types
interface Message {
  id: string;
  type: 'bot' | 'user';
  content: string;
  quickReplies?: string[];
  timestamp: Date;
}

// Find best matching response
function findResponse(input: string): KnowledgeEntry {
  const lowerInput = input.toLowerCase();
  
  for (const entry of KNOWLEDGE_BASE) {
    for (const keyword of entry.keywords) {
      if (lowerInput.includes(keyword.toLowerCase())) {
        return entry;
      }
    }
  }
  
  return DEFAULT_RESPONSE;
}

// Generate unique ID
function generateId(): string {
  return Math.random().toString(36).substr(2, 9);
}

export default function EyeBot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: generateId(),
      type: 'bot',
      content: KNOWLEDGE_BASE[0].response,
      quickReplies: KNOWLEDGE_BASE[0].quickReplies,
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 300);
    }
  }, [isOpen]);

  const handleSend = (text: string = inputValue) => {
    if (!text.trim()) return;

    // Add user message
    const userMessage: Message = {
      id: generateId(),
      type: 'user',
      content: text,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    // Simulate AI response with delay
    setTimeout(() => {
      const response = findResponse(text);
      const botMessage: Message = {
        id: generateId(),
        type: 'bot',
        content: response.response,
        quickReplies: response.quickReplies,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, botMessage]);
      setIsTyping(false);
    }, 600 + Math.random() * 400);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleQuickReply = (reply: string) => {
    handleSend(reply);
  };

  const handleClearChat = () => {
    setMessages([
      {
        id: generateId(),
        type: 'bot',
        content: KNOWLEDGE_BASE[0].response,
        quickReplies: KNOWLEDGE_BASE[0].quickReplies,
        timestamp: new Date(),
      },
    ]);
  };

  const renderMarkdownLine = (line: string) => {
    // Basic Markdown parser for **bold** text inline
    const parts = line.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, index) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={index} className="font-semibold text-foreground">{part.slice(2, -2)}</strong>;
      }
      return part;
    });
  };

  return (
    <>
      {/* Floating Button */}
      <AnimatePresence>
        {!isOpen && (
          <motion.button
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setIsOpen(true)}
            className="fixed bottom-6 right-6 z-50 w-14 h-14 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full shadow-lg flex items-center justify-center group"
          >
            <MessageCircle className="w-6 h-6 text-white" />
            <span className="absolute -top-1 -right-1 w-4 h-4 bg-emerald-500 rounded-full border-2 border-background" />
            
            {/* Tooltip */}
            <div className="absolute right-full mr-3 px-3 py-1.5 bg-card border border-border rounded-lg shadow-lg whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity">
              <span className="text-sm font-medium">Ask EyeBot</span>
            </div>
          </motion.button>
        )}
      </AnimatePresence>

      {/* Chat Window */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ duration: 0.3, type: 'spring' }}
            className="fixed bottom-6 right-6 z-50 w-[400px] max-w-[calc(100vw-48px)] h-[600px] max-h-[calc(100vh-100px)] bg-card/95 backdrop-blur-xl border border-border rounded-2xl shadow-2xl flex flex-col overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-border bg-gradient-to-r from-cyan-500/10 to-blue-500/10">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center">
                  <Bot className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h3 className="font-semibold text-sm">EyeBot</h3>
                  <p className="text-xs text-muted-foreground flex items-center gap-1">
                    <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full" />
                    AI Assistant
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-muted-foreground hover:text-red-500 hover:bg-red-500/10"
                  onClick={handleClearChat}
                  title="Clear Chat"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  onClick={() => setIsOpen(false)}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </div>

            {/* Messages — scrollable */}
            <div className="flex-1 overflow-y-auto p-4" style={{ scrollBehavior: 'smooth' }}>
              <div className="space-y-4">
                {messages.map((message) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <div
                      className={`flex gap-2 ${
                        message.type === 'user' ? 'flex-row-reverse' : ''
                      }`}
                    >
                      <div
                        className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${
                          message.type === 'bot'
                            ? 'bg-gradient-to-br from-cyan-500 to-blue-600'
                            : 'bg-muted'
                        }`}
                      >
                        {message.type === 'bot' ? (
                          <Sparkles className="w-3.5 h-3.5 text-white" />
                        ) : (
                          <User className="w-3.5 h-3.5" />
                        )}
                      </div>
                      <div
                        className={`max-w-[80%] p-3 rounded-2xl text-sm leading-relaxed ${
                          message.type === 'bot'
                            ? 'bg-muted rounded-tl-sm'
                            : 'bg-primary text-primary-foreground rounded-tr-sm'
                        }`}
                      >
                        {/* Parse markdown-like formatting */}
                        {message.content.split('\n').map((line, i) => (
                          <span key={i} className="block min-h-[1.2em]">
                            {line.startsWith('•') || line.startsWith('-') ? (
                              <span className="flex items-start gap-2">
                                <span className="mt-1.5 w-1 h-1 bg-current rounded-full flex-shrink-0" />
                                <span>{renderMarkdownLine(line.substring(2))}</span>
                              </span>
                            ) : line.startsWith('📋') || line.startsWith('📊') || line.startsWith('🔍') || line.startsWith('📅') || line.startsWith('💡') || line.startsWith('📖') ? (
                              <span className="flex items-start gap-2 mt-1">
                                <span>{line.substring(0, 2)}</span>
                                <span>{renderMarkdownLine(line.substring(2))}</span>
                              </span>
                            ) : (
                              renderMarkdownLine(line)
                            )}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Quick Replies */}
                    {message.quickReplies && message.type === 'bot' && (
                      <div className="flex flex-wrap gap-2 mt-2 ml-9">
                        {message.quickReplies.map((reply) => (
                          <button
                            key={reply}
                            onClick={() => handleQuickReply(reply)}
                            className="px-3 py-1.5 bg-primary/10 hover:bg-primary/20 border border-primary/20 rounded-full text-xs font-medium transition-colors flex items-center gap-1"
                          >
                            {reply}
                            <ChevronRight className="w-3 h-3" />
                          </button>
                        ))}
                      </div>
                    )}
                  </motion.div>
                ))}

                {/* Typing Indicator */}
                {isTyping && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex items-center gap-2 ml-9"
                  >
                    <div className="w-7 h-7 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center">
                      <Sparkles className="w-3.5 h-3.5 text-white" />
                    </div>
                    <div className="bg-muted px-4 py-2 rounded-2xl rounded-tl-sm flex items-center gap-1">
                      <motion.div
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{ duration: 0.6, repeat: Infinity }}
                        className="w-2 h-2 bg-muted-foreground rounded-full"
                      />
                      <motion.div
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{ duration: 0.6, repeat: Infinity, delay: 0.2 }}
                        className="w-2 h-2 bg-muted-foreground rounded-full"
                      />
                      <motion.div
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{ duration: 0.6, repeat: Infinity, delay: 0.4 }}
                        className="w-2 h-2 bg-muted-foreground rounded-full"
                      />
                    </div>
                  </motion.div>
                )}

                <div ref={messagesEndRef} />
              </div>
            </div>

            {/* Input */}
            <div className="p-4 border-t border-border">
              <div className="flex items-center gap-2">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask about eye diseases, screening..."
                  className="flex-1 px-4 py-2.5 bg-muted rounded-full text-sm border border-transparent focus:border-primary focus:outline-none transition-colors"
                />
                <Button
                  size="icon"
                  className="h-10 w-10 rounded-full btn-medical"
                  onClick={() => handleSend()}
                  disabled={!inputValue.trim() || isTyping}
                >
                  <Send className="w-4 h-4" />
                </Button>
              </div>
              <p className="text-[10px] text-muted-foreground text-center mt-2">
                EyeBot provides general information. Always consult a doctor for medical advice.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

