package com.hismith.hismithapp.manager;

import android.content.Context;
import android.media.AudioRecord;
import android.media.audiofx.AcousticEchoCanceler;
import android.media.audiofx.NoiseSuppressor;
import android.util.Base64;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.audio.RTPPackage;
import com.hismith.hismithapp.audio.TrackBase;
import com.hismith.hismithapp.utils.AudioUtil;
import com.hismith.hismithapp.utils.ByteUtil;
import com.hismith.hismithapp.utils.LogUtil;
import java.util.LinkedList;

/* loaded from: classes3.dex */
public class LineAudioManager {
    private static LineAudioManager instance;
    private AcousticEchoCanceler mAcousticEchoCanceler;
    private AudioRecord mAudioRecord;
    private final Context mContext;
    private int mInBuffSize;
    private byte[] mInBytes;
    private LinkedList<byte[]> mInQList;
    private boolean mInRunning;
    private boolean mInSpeed;
    private NoiseSuppressor mNoiseSuppressor;
    protected int mOutBuffSize;
    protected byte[] mOutBytes;
    private Thread mRecordThread;
    private Thread mSpeedThread;
    private OnAudioSpeedChangeListener onAudioSpeedChangeListener;
    private TrackBase trackBase;
    private final int mSampleRateInHz = 8000;
    private final int mPcmEncoding = 2;
    private final int mWriteSize = 1600;
    private int mSessionId = -1;
    private boolean isTrackBaseBegin = false;
    private byte[] mRecordBytes = new byte[0];
    private String isSocketMic = "0";
    Runnable recordRunnable = new Runnable() { // from class: com.hismith.hismithapp.manager.LineAudioManager.1
        @Override // java.lang.Runnable
        public void run() {
            LineAudioManager lineAudioManager;
            try {
                try {
                    LineAudioManager.this.mAudioRecord.startRecording();
                    while (true) {
                        boolean z = LineAudioManager.this.mInRunning;
                        lineAudioManager = LineAudioManager.this;
                        if (!z) {
                            break;
                        }
                        int read = lineAudioManager.mAudioRecord.read(LineAudioManager.this.mInBytes, 0, LineAudioManager.this.mInBuffSize);
                        if (read != -3 && read != -2) {
                            if (read > 0) {
                                LineAudioManager lineAudioManager2 = LineAudioManager.this;
                                lineAudioManager2.mRecordBytes = ByteUtil.mergeArr(lineAudioManager2.mRecordBytes, LineAudioManager.this.mInBytes);
                                if (LineAudioManager.this.mRecordBytes.length >= 1600) {
                                    byte[] bArr = new byte[1600];
                                    System.arraycopy(LineAudioManager.this.mRecordBytes, 0, bArr, 0, 1600);
                                    LineAudioManager lineAudioManager3 = LineAudioManager.this;
                                    lineAudioManager3.mRecordBytes = ByteUtil.remoteArr(lineAudioManager3.mRecordBytes, 1600);
                                    boolean isMute = AudioUtil.isMute(bArr, 16);
                                    LogUtil.e("isMute:" + isMute + ", isSocketMic:" + LineAudioManager.this.isSocketMic);
                                    if (!isMute && !"0".equals(LineAudioManager.this.isSocketMic)) {
                                        String encodeToString = Base64.encodeToString(bArr, 2);
                                        LogUtil.e("data.length:1600, mRecordBytes:" + LineAudioManager.this.mRecordBytes.length + ", readSize:" + read);
                                        AudioSocketManager.getInstance().roomVoice(encodeToString);
                                    }
                                }
                            }
                        }
                        LogUtil.e("Could not read audio data.");
                    }
                    lineAudioManager.mAudioRecord.stop();
                    LineAudioManager.this.mRecordBytes = new byte[0];
                } catch (Exception e) {
                    e.printStackTrace();
                    LineAudioManager.this.releaseAudio();
                }
            } finally {
                LineAudioManager.this.mRecordThread = null;
            }
        }
    };
    private int maxVolume = 85;
    private int minVolume = 30;
    private double lastVolume = 0.0d;
    private final byte[] mSpeedByteArr = new byte[0];
    Runnable speedRunnable = new Runnable() { // from class: com.hismith.hismithapp.manager.LineAudioManager.2
        @Override // java.lang.Runnable
        public void run() {
            LineAudioManager lineAudioManager;
            try {
                try {
                    LineAudioManager.this.mAudioRecord.startRecording();
                    LineAudioManager.this.mInSpeed = true;
                    while (true) {
                        boolean z = LineAudioManager.this.mInSpeed;
                        lineAudioManager = LineAudioManager.this;
                        if (!z) {
                            break;
                        }
                        int read = lineAudioManager.mAudioRecord.read(LineAudioManager.this.mInBytes, 0, LineAudioManager.this.mInBuffSize);
                        if (read != -3 && read != -2) {
                            LineAudioManager lineAudioManager2 = LineAudioManager.this;
                            double calculateVolume = lineAudioManager2.calculateVolume(lineAudioManager2.mInBytes);
                            LogUtil.e("分贝：" + calculateVolume);
                            if (calculateVolume <= LineAudioManager.this.lastVolume + 10.0d) {
                                double unused = LineAudioManager.this.lastVolume;
                            }
                            LineAudioManager.this.lastVolume = calculateVolume;
                            double d = calculateVolume - LineAudioManager.this.minVolume;
                            if (d < 0.0d) {
                                d = 0.0d;
                            }
                            int i = (int) ((d / (LineAudioManager.this.maxVolume - LineAudioManager.this.minVolume)) * 100.0d);
                            if (i > 100) {
                                i = 100;
                            }
                            if (LineAudioManager.this.onAudioSpeedChangeListener != null) {
                                LineAudioManager.this.onAudioSpeedChangeListener.onAudioSpeedChange(LineAudioManager.this.lastVolume, i);
                            }
                            try {
                                Thread.sleep(100L);
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }
                        LogUtil.e("Could not read audio data.");
                    }
                    lineAudioManager.mAudioRecord.stop();
                } catch (Exception e2) {
                    e2.printStackTrace();
                    LineAudioManager.this.releaseAudio();
                }
            } finally {
                LineAudioManager.this.mSpeedThread = null;
            }
        }
    };

    public interface OnAudioSpeedChangeListener {
        void onAudioSpeedChange(double d, int i);
    }

    public LineAudioManager(Context context) {
        this.mContext = context;
        initRecord();
        TrackBase trackBase = new TrackBase();
        this.trackBase = trackBase;
        trackBase.start();
    }

    private boolean addIncoming(String str, String str2) {
        return this.trackBase.addPacket(new RTPPackage(str, str2));
    }

    /* JADX INFO: Access modifiers changed from: private */
    public double calculateVolume(byte[] bArr) {
        double d = 0.0d;
        for (int i = 0; i < bArr.length; i += 2) {
            int i2 = (bArr[i] & 255) + ((bArr[i + 1] & 255) << 8);
            if (i2 >= 32768) {
                i2 = 65535 - i2;
            }
            d += Math.abs(i2);
        }
        return Math.log10(((d / bArr.length) / 2.0d) + 1.0d) * 20.0d;
    }

    public static LineAudioManager getInstance() {
        LineAudioManager lineAudioManager;
        synchronized (LineAudioManager.class) {
            lineAudioManager = instance;
            if (lineAudioManager == null) {
                lineAudioManager = new LineAudioManager(MyApplication.getApplication());
                instance = lineAudioManager;
            }
        }
        return lineAudioManager;
    }

    public String getIsSocketMic() {
        return this.isSocketMic;
    }

    public int getMaxVolume() {
        return this.maxVolume;
    }

    public int getMinVolume() {
        return this.minVolume;
    }

    public void initRecord() {
        try {
            this.mInBuffSize = AudioRecord.getMinBufferSize(8000, 16, 2);
            this.mAudioRecord = new AudioRecord(7, 8000, 16, 2, this.mInBuffSize);
            this.mInBytes = new byte[this.mInBuffSize];
            this.mInQList = new LinkedList<>();
            this.mSessionId = this.mAudioRecord.getAudioSessionId();
            if (AcousticEchoCanceler.isAvailable()) {
                AcousticEchoCanceler create = AcousticEchoCanceler.create(this.mSessionId);
                this.mAcousticEchoCanceler = create;
                if (create != null) {
                    create.setEnabled(true);
                }
            }
            if (NoiseSuppressor.isAvailable()) {
                NoiseSuppressor create2 = NoiseSuppressor.create(this.mSessionId);
                this.mNoiseSuppressor = create2;
                if (create2 != null) {
                    create2.setEnabled(true);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void playAudio(String str, String str2) {
        try {
            if (!addIncoming(str, str2)) {
                LogUtil.e("addIncoming : Failed !");
            }
            if (this.isTrackBaseBegin) {
                return;
            }
            this.trackBase.begin();
            this.isTrackBaseBegin = true;
        } catch (Exception e) {
            e.printStackTrace();
            LogUtil.e("base64解析错误");
        }
    }

    public void release() {
        releaseAudio();
        releaseAudioTrack();
        if (instance != null) {
            instance = null;
        }
    }

    public void releaseAudio() {
        AudioRecord audioRecord = this.mAudioRecord;
        if (audioRecord != null) {
            audioRecord.release();
            this.mAudioRecord = null;
        }
    }

    public void releaseAudioTrack() {
        try {
            TrackBase trackBase = this.trackBase;
            if (trackBase != null) {
                trackBase.release();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void setIsSocketMic(String str) {
        this.isSocketMic = str;
    }

    public void setMaxVolume(int i) {
        this.maxVolume = i;
    }

    public void setMinVolume(int i) {
        this.minVolume = i;
    }

    public void setOnAudioSpeedChangeListener(OnAudioSpeedChangeListener onAudioSpeedChangeListener) {
        this.onAudioSpeedChangeListener = onAudioSpeedChangeListener;
    }

    public void startAudio() {
        if (this.mInRunning) {
            return;
        }
        if (this.mRecordThread != null) {
            this.mRecordThread = null;
        }
        this.mInRunning = true;
        Thread thread = new Thread(this.recordRunnable);
        this.mRecordThread = thread;
        thread.start();
        LogUtil.e("开始录音");
    }

    public void startSpeedAudio() {
        if (this.mSpeedThread != null) {
            this.mSpeedThread = null;
        }
        Thread thread = new Thread(this.speedRunnable);
        this.mSpeedThread = thread;
        thread.start();
    }

    public void stopAudio() {
        if (this.mInRunning) {
            this.mInRunning = false;
        }
    }

    public void stopAudioTrack() {
        TrackBase trackBase = this.trackBase;
        if (trackBase != null) {
            trackBase.over();
        }
    }

    public void stopSpeedAudio() {
        this.mInSpeed = false;
        setOnAudioSpeedChangeListener(null);
    }
}
