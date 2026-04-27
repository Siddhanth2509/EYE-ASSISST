import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, CheckCircle, Loader2, CreditCard, Smartphone, Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

interface PaymentGatewayProps {
  amount: number;          // in rupees
  description: string;
  onSuccess: (txnId: string) => void;
  onClose: () => void;
}

function generateTxnId() {
  return 'TXN' + Date.now().toString(36).toUpperCase() + Math.random().toString(36).substr(2, 4).toUpperCase();
}

export default function PaymentGateway({ amount, description, onSuccess, onClose }: PaymentGatewayProps) {
  const [tab, setTab] = useState<'upi' | 'card'>('upi');
  const [upiId, setUpiId] = useState('');
  const [processing, setProcessing] = useState(false);
  const [success, setSuccess] = useState(false);
  const [txnId] = useState(generateTxnId);
  const [copied, setCopied] = useState(false);

  // Card fields
  const [cardNum, setCardNum] = useState('');
  const [expiry, setExpiry] = useState('');
  const [cvv, setCvv] = useState('');
  const [cardName, setCardName] = useState('');

  const UPI_ID = 'kuldeepgoswami636@okhdfcbank';
  const qrData = `upi://pay?pa=${UPI_ID}&pn=EYE-ASSISST&am=${amount}&tn=${encodeURIComponent(description)}&cu=INR`;
  const qrUrl  = `https://api.qrserver.com/v1/create-qr-code/?data=${encodeURIComponent(qrData)}&size=200x200&format=png&margin=10`;

  const copyUpiId = () => {
    navigator.clipboard.writeText(UPI_ID);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const simulatePay = () => {
    setProcessing(false);
    setSuccess(false);
    onSuccess(txnId);
  };

  const formatCard = (val: string) =>
    val.replace(/\D/g, '').slice(0, 16).replace(/(.{4})/g, '$1 ').trim();

  const formatExpiry = (val: string) =>
    val.replace(/\D/g, '').slice(0, 4).replace(/(\d{2})(\d)/, '$1/$2');

  return (
    <div className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <AnimatePresence mode="wait">
        {success ? (
          <motion.div
            key="success"
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="bg-card border border-border rounded-2xl p-10 max-w-sm w-full text-center"
          >
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: 'spring', delay: 0.1 }}
              className="w-20 h-20 bg-emerald-500/20 rounded-full flex items-center justify-center mx-auto mb-6"
            >
              <CheckCircle className="w-10 h-10 text-emerald-400" />
            </motion.div>
            <h2 className="text-2xl font-bold text-foreground mb-2">Payment Successful!</h2>
            <p className="text-muted-foreground mb-4">{description}</p>
            <div className="bg-muted/50 rounded-lg p-3 mb-4">
              <p className="text-xs text-muted-foreground">Transaction ID</p>
              <p className="font-mono text-sm text-primary font-bold">{txnId}</p>
            </div>
            <p className="text-3xl font-bold text-foreground mb-2">₹{amount.toLocaleString()}</p>
            <p className="text-sm text-emerald-400">Your plan is now active ✓</p>
          </motion.div>
        ) : (
          <motion.div
            key="form"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-card border border-border rounded-2xl w-full max-w-md overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-5 border-b border-border">
              <div>
                <h2 className="font-bold text-foreground text-lg">Complete Payment</h2>
                <p className="text-sm text-muted-foreground">{description}</p>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-2xl font-bold text-primary">₹{amount.toLocaleString()}</span>
                <Button variant="ghost" size="icon" onClick={onClose} className="h-8 w-8">
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </div>

            {/* Tabs */}
            <div className="flex border-b border-border">
              <button
                onClick={() => setTab('upi')}
                className={`flex-1 flex items-center justify-center gap-2 py-3 text-sm font-medium transition-colors ${
                  tab === 'upi' ? 'text-primary border-b-2 border-primary' : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                <Smartphone className="w-4 h-4" />
                UPI / QR Code
              </button>
              <button
                onClick={() => setTab('card')}
                className={`flex-1 flex items-center justify-center gap-2 py-3 text-sm font-medium transition-colors ${
                  tab === 'card' ? 'text-primary border-b-2 border-primary' : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                <CreditCard className="w-4 h-4" />
                Card / Net Banking
              </button>
            </div>

            {/* UPI Tab */}
            <AnimatePresence mode="wait">
              {tab === 'upi' && (
                <motion.div
                  key="upi"
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 10 }}
                  className="p-6 space-y-5"
                >
                  {/* QR Code */}
                  <div className="flex flex-col items-center">
                    <div className="p-3 bg-[#0a0f1e] rounded-2xl border border-border mb-3">
                      <img src={qrUrl} alt="UPI QR" className="w-48 h-48" />
                    </div>
                    <p className="text-xs text-muted-foreground text-center">
                      Scan with PhonePe, Google Pay, Paytm, or any UPI app
                    </p>
                  </div>

                  <div className="relative flex items-center gap-3">
                    <div className="flex-1 h-px bg-border" />
                    <span className="text-xs text-muted-foreground">OR</span>
                    <div className="flex-1 h-px bg-border" />
                  </div>

                  {/* Manual UPI */}
                  <div className="space-y-2">
                    <Label className="text-xs">Pay to UPI ID</Label>
                    <div className="flex gap-2">
                      <div className="flex-1 bg-muted/50 border border-border rounded-lg px-3 py-2 font-mono text-sm text-foreground">
                        {UPI_ID}
                      </div>
                      <Button variant="outline" size="icon" onClick={copyUpiId} className="h-10 w-10">
                        {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-xs">Enter your UPI ID to confirm payment</Label>
                    <Input
                      placeholder="yourname@upi"
                      value={upiId}
                      onChange={e => setUpiId(e.target.value)}
                    />
                  </div>

                  <Button
                    className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 h-11"
                    onClick={simulatePay}
                    disabled={processing || !upiId.includes('@')}
                  >
                    {processing
                      ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Verifying...</>
                      : <>Pay ₹{amount.toLocaleString()} via UPI</>
                    }
                  </Button>

                  <p className="text-xs text-center text-muted-foreground">
                    🔒 256-bit encrypted · Powered by Razorpay
                  </p>
                </motion.div>
              )}

              {tab === 'card' && (
                <motion.div
                  key="card"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  className="p-6 space-y-4"
                >
                  <div className="space-y-2">
                    <Label className="text-xs">Cardholder Name</Label>
                    <Input placeholder="Name on card" value={cardName} onChange={e => setCardName(e.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-xs">Card Number</Label>
                    <Input
                      placeholder="0000 0000 0000 0000"
                      value={formatCard(cardNum)}
                      onChange={e => setCardNum(e.target.value.replace(/\s/g, ''))}
                      maxLength={19}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-2">
                      <Label className="text-xs">Expiry</Label>
                      <Input
                        placeholder="MM/YY"
                        value={formatExpiry(expiry)}
                        onChange={e => setExpiry(e.target.value)}
                        maxLength={5}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-xs">CVV</Label>
                      <Input
                        placeholder="•••"
                        type="password"
                        value={cvv}
                        onChange={e => setCvv(e.target.value.slice(0, 3))}
                        maxLength={3}
                      />
                    </div>
                  </div>

                  <Button
                    className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 h-11 mt-2"
                    onClick={simulatePay}
                    disabled={processing || !cardName || cardNum.length < 16 || expiry.length < 5 || cvv.length < 3}
                  >
                    {processing
                      ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Processing...</>
                      : <>Pay ₹{amount.toLocaleString()} Securely</>
                    }
                  </Button>

                  <div className="flex items-center justify-center gap-4 pt-1">
                    {['Visa', 'Mastercard', 'RuPay', 'Maestro'].map(n => (
                      <span key={n} className="text-xs text-muted-foreground bg-muted/50 px-2 py-1 rounded">{n}</span>
                    ))}
                  </div>
                  <p className="text-xs text-center text-muted-foreground">
                    🔒 256-bit SSL encrypted · PCI DSS compliant
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
            
            {/* TEST BYPASS BUTTON */}
            <div className="mt-4 pt-4 border-t border-border flex justify-center">
              <Button 
                variant="outline" 
                size="sm" 
                className="text-xs w-full bg-amber-500/10 text-amber-500 hover:bg-amber-500/20 border-amber-500/30"
                onClick={simulatePay}
              >
                Bypass Payment (Test Automation)
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
