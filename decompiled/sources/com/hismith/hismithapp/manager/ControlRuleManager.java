package com.hismith.hismithapp.manager;

import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.text.TextUtils;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.bluetooth.BleProduct;
import com.hismith.hismithapp.device_rule.ConditionType;
import com.hismith.hismithapp.device_rule.SpeedChannel;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/* loaded from: classes3.dex */
public class ControlRuleManager {
    private static ControlRuleManager instance;
    private Map<String, ControlRule> deviceRules = new HashMap();
    private Gson gson = new Gson();
    private String leaderId;

    /* renamed from: com.hismith.hismithapp.manager.ControlRuleManager$2, reason: invalid class name */
    static /* synthetic */ class AnonymousClass2 {
        static final /* synthetic */ int[] $SwitchMap$com$hismith$hismithapp$device_rule$ConditionType;

        static {
            int[] iArr = new int[ConditionType.values().length];
            $SwitchMap$com$hismith$hismithapp$device_rule$ConditionType = iArr;
            try {
                iArr[ConditionType.GREATER_THAN.ordinal()] = 1;
            } catch (NoSuchFieldError unused) {
            }
            try {
                $SwitchMap$com$hismith$hismithapp$device_rule$ConditionType[ConditionType.EQUAL.ordinal()] = 2;
            } catch (NoSuchFieldError unused2) {
            }
            try {
                $SwitchMap$com$hismith$hismithapp$device_rule$ConditionType[ConditionType.LESS_THAN.ordinal()] = 3;
            } catch (NoSuchFieldError unused3) {
            }
        }
    }

    public static class ControlRule implements Cloneable {
        private final int WHAT_DELAY_PIU;
        private final int WHAT_DURATION_A;
        private final int WHAT_DURATION_B;
        private ConditionType condition;
        private int durationA;
        private long durationAEndTime;
        private int durationB;
        private long durationBEndTime;
        private transient boolean enabled;
        private float factorA;
        private float factorB;
        private boolean followA;
        private boolean followB;
        private String followerId;
        private transient boolean isEligible;
        private transient Handler mHandler;
        private transient ProductModel mProductModel;
        private transient int maxLimit;
        private transient int minLimit;
        private int offValueA;
        private int offValueB;
        private int onValueA;
        private int onValueB;
        private SpeedChannel speedChannel;
        private int switchDelay;
        private int threshold;
        private boolean triggerState;
        private SpeedChannel vibrateChannel;

        public ControlRule(String str, float f, float f2, SpeedChannel speedChannel, SpeedChannel speedChannel2, ProductModel productModel) {
            this.factorA = 1.0f;
            this.factorB = 1.0f;
            this.speedChannel = SpeedChannel.CHANNEL_A;
            this.vibrateChannel = SpeedChannel.CHANNEL_B;
            this.minLimit = 0;
            this.maxLimit = 100;
            this.enabled = true;
            this.condition = ConditionType.EQUAL;
            this.threshold = 50;
            this.triggerState = true;
            this.onValueA = 50;
            this.offValueA = 0;
            this.durationA = 0;
            this.followA = true;
            this.onValueB = 50;
            this.offValueB = 0;
            this.durationB = 0;
            this.followB = true;
            this.switchDelay = 0;
            this.isEligible = false;
            this.durationAEndTime = 0L;
            this.durationBEndTime = 0L;
            this.WHAT_DURATION_A = 1;
            this.WHAT_DURATION_B = 2;
            this.WHAT_DELAY_PIU = 3;
            this.followerId = str;
            setFactorA(f);
            setFactorB(f2);
            setSpeedChannel(speedChannel);
            setVibrateChannel(speedChannel2);
            this.mProductModel = productModel;
        }

        public ControlRule(String str, float f, float f2, ProductModel productModel) {
            this.factorA = 1.0f;
            this.factorB = 1.0f;
            this.speedChannel = SpeedChannel.CHANNEL_A;
            this.vibrateChannel = SpeedChannel.CHANNEL_B;
            this.minLimit = 0;
            this.maxLimit = 100;
            this.enabled = true;
            this.condition = ConditionType.EQUAL;
            this.threshold = 50;
            this.triggerState = true;
            this.onValueA = 50;
            this.offValueA = 0;
            this.durationA = 0;
            this.followA = true;
            this.onValueB = 50;
            this.offValueB = 0;
            this.durationB = 0;
            this.followB = true;
            this.switchDelay = 0;
            this.isEligible = false;
            this.durationAEndTime = 0L;
            this.durationBEndTime = 0L;
            this.WHAT_DURATION_A = 1;
            this.WHAT_DURATION_B = 2;
            this.WHAT_DELAY_PIU = 3;
            this.followerId = str;
            setFactorA(f);
            setFactorB(f2);
            this.mProductModel = productModel;
        }

        public ControlRule(String str, int i, ProductModel productModel) {
            this.factorA = 1.0f;
            this.factorB = 1.0f;
            this.speedChannel = SpeedChannel.CHANNEL_A;
            this.vibrateChannel = SpeedChannel.CHANNEL_B;
            this.minLimit = 0;
            this.maxLimit = 100;
            this.enabled = true;
            this.condition = ConditionType.EQUAL;
            this.threshold = 50;
            this.triggerState = true;
            this.onValueA = 50;
            this.offValueA = 0;
            this.durationA = 0;
            this.followA = true;
            this.onValueB = 50;
            this.offValueB = 0;
            this.durationB = 0;
            this.followB = true;
            this.isEligible = false;
            this.durationAEndTime = 0L;
            this.durationBEndTime = 0L;
            this.WHAT_DURATION_A = 1;
            this.WHAT_DURATION_B = 2;
            this.WHAT_DELAY_PIU = 3;
            this.followerId = str;
            this.switchDelay = i;
            this.mProductModel = productModel;
        }

        public ControlRule(String str, ConditionType conditionType, int i, boolean z, ProductModel productModel) {
            this.factorA = 1.0f;
            this.factorB = 1.0f;
            this.speedChannel = SpeedChannel.CHANNEL_A;
            this.vibrateChannel = SpeedChannel.CHANNEL_B;
            this.minLimit = 0;
            this.maxLimit = 100;
            this.enabled = true;
            ConditionType conditionType2 = ConditionType.EQUAL;
            this.onValueA = 50;
            this.offValueA = 0;
            this.durationA = 0;
            this.followA = true;
            this.onValueB = 50;
            this.offValueB = 0;
            this.durationB = 0;
            this.followB = true;
            this.switchDelay = 0;
            this.isEligible = false;
            this.durationAEndTime = 0L;
            this.durationBEndTime = 0L;
            this.WHAT_DURATION_A = 1;
            this.WHAT_DURATION_B = 2;
            this.WHAT_DELAY_PIU = 3;
            this.followerId = str;
            this.condition = conditionType;
            this.threshold = i;
            this.triggerState = z;
            this.mProductModel = productModel;
        }

        public ControlRule(String str, ProductModel productModel, int i, int i2) {
            this.factorA = 1.0f;
            this.factorB = 1.0f;
            this.speedChannel = SpeedChannel.CHANNEL_A;
            this.vibrateChannel = SpeedChannel.CHANNEL_B;
            this.minLimit = 0;
            this.maxLimit = 100;
            this.enabled = true;
            this.condition = ConditionType.EQUAL;
            this.threshold = 50;
            this.triggerState = true;
            this.durationA = 0;
            this.onValueB = 50;
            this.offValueB = 0;
            this.durationB = 0;
            this.switchDelay = 0;
            this.isEligible = false;
            this.durationAEndTime = 0L;
            this.durationBEndTime = 0L;
            this.WHAT_DURATION_A = 1;
            this.WHAT_DURATION_B = 2;
            this.WHAT_DELAY_PIU = 3;
            this.followerId = str;
            this.mProductModel = productModel;
            this.onValueA = i;
            this.offValueA = i2;
            this.followA = true;
            this.followB = false;
        }

        public ControlRule(String str, ProductModel productModel, int i, int i2, int i3, int i4) {
            this.factorA = 1.0f;
            this.factorB = 1.0f;
            this.speedChannel = SpeedChannel.CHANNEL_A;
            this.vibrateChannel = SpeedChannel.CHANNEL_B;
            this.minLimit = 0;
            this.maxLimit = 100;
            this.enabled = true;
            this.condition = ConditionType.EQUAL;
            this.threshold = 50;
            this.triggerState = true;
            this.durationA = 0;
            this.durationB = 0;
            this.switchDelay = 0;
            this.isEligible = false;
            this.durationAEndTime = 0L;
            this.durationBEndTime = 0L;
            this.WHAT_DURATION_A = 1;
            this.WHAT_DURATION_B = 2;
            this.WHAT_DELAY_PIU = 3;
            this.followerId = str;
            this.mProductModel = productModel;
            this.onValueA = i;
            this.offValueA = i2;
            this.followA = true;
            this.onValueB = i3;
            this.offValueB = i4;
            this.followB = true;
        }

        public ControlRule(String str, boolean z) {
            this.factorA = 1.0f;
            this.factorB = 1.0f;
            this.speedChannel = SpeedChannel.CHANNEL_A;
            this.vibrateChannel = SpeedChannel.CHANNEL_B;
            this.minLimit = 0;
            this.maxLimit = 100;
            this.enabled = true;
            this.condition = ConditionType.EQUAL;
            this.threshold = 50;
            this.triggerState = true;
            this.onValueA = 50;
            this.offValueA = 0;
            this.durationA = 0;
            this.followA = true;
            this.onValueB = 50;
            this.offValueB = 0;
            this.durationB = 0;
            this.followB = true;
            this.switchDelay = 0;
            this.isEligible = false;
            this.durationAEndTime = 0L;
            this.durationBEndTime = 0L;
            this.WHAT_DURATION_A = 1;
            this.WHAT_DURATION_B = 2;
            this.WHAT_DELAY_PIU = 3;
            this.followerId = str;
            this.enabled = z;
        }

        public ControlRule(String str, boolean z, ProductModel productModel) {
            this.factorA = 1.0f;
            this.factorB = 1.0f;
            this.speedChannel = SpeedChannel.CHANNEL_A;
            this.vibrateChannel = SpeedChannel.CHANNEL_B;
            this.minLimit = 0;
            this.maxLimit = 100;
            this.enabled = true;
            this.condition = ConditionType.EQUAL;
            this.threshold = 50;
            this.onValueA = 50;
            this.offValueA = 0;
            this.durationA = 0;
            this.followA = true;
            this.onValueB = 50;
            this.offValueB = 0;
            this.durationB = 0;
            this.followB = true;
            this.switchDelay = 0;
            this.isEligible = false;
            this.durationAEndTime = 0L;
            this.durationBEndTime = 0L;
            this.WHAT_DURATION_A = 1;
            this.WHAT_DURATION_B = 2;
            this.WHAT_DELAY_PIU = 3;
            this.followerId = str;
            this.triggerState = z;
            this.mProductModel = productModel;
        }

        private boolean evaluateCondition(int i, int i2) {
            int i3 = AnonymousClass2.$SwitchMap$com$hismith$hismithapp$device_rule$ConditionType[this.condition.ordinal()];
            return i3 != 1 ? i3 != 2 ? i3 == 3 && i < i2 : i == i2 : i > i2;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public Handler getOrCreateHandler() {
            Handler handler = this.mHandler;
            if (handler != null) {
                return handler;
            }
            Handler handler2 = new Handler(Looper.getMainLooper(), new Handler.Callback() { // from class: com.hismith.hismithapp.manager.ControlRuleManager.ControlRule.1
                @Override // android.os.Handler.Callback
                public boolean handleMessage(Message message) {
                    int i = message.what;
                    BleProduct connBleProduct = BleManager.getInstance(MyApplication.getApplication()).getConnBleProduct(ControlRule.this.followerId);
                    if (i == 1) {
                        LogUtil.e("applyRuleSwitch handleMessage:WHAT_DURATION_A writeSpeed(0)");
                        if (connBleProduct != null) {
                            connBleProduct.writeSpeed(0);
                        }
                        ControlRule.this.durationAEndTime = 0L;
                    } else if (i == 2) {
                        if (connBleProduct != null) {
                            connBleProduct.writeVibrate(0);
                        }
                        ControlRule.this.durationBEndTime = 0L;
                    } else if (i == 3 && connBleProduct != null) {
                        connBleProduct.writeOpenPiu();
                    }
                    return false;
                }
            });
            this.mHandler = handler2;
            return handler2;
        }

        public void applyRule(int i, SpeedChannel speedChannel) {
            ProductModel productModel;
            if (!this.enabled || (productModel = this.mProductModel) == null) {
                LogUtil.e("applyRule enabled:" + this.enabled + ", mProductModel:" + this.mProductModel);
                return;
            }
            boolean equals = Constans.DEVICE_TYPE_FUN.equals(productModel.getDeviceType());
            SpeedChannel speedChannel2 = this.speedChannel;
            if (equals) {
                if (speedChannel2 == speedChannel) {
                    boolean evaluateCondition = evaluateCondition(i, this.threshold);
                    LogUtil.e("主速度，从触发：input：" + i + ", threshold:" + this.threshold + ", shouldTrigger:" + evaluateCondition);
                    if (!evaluateCondition) {
                        this.isEligible = false;
                        return;
                    }
                    if (this.isEligible) {
                        return;
                    }
                    this.isEligible = true;
                    BleProduct connBleProduct = BleManager.getInstance(MyApplication.getApplication()).getConnBleProduct(this.followerId);
                    if (connBleProduct != null) {
                        connBleProduct.writeOpenPiu();
                        return;
                    }
                    return;
                }
                return;
            }
            if (speedChannel2 == speedChannel) {
                int i2 = (int) (i * this.factorA);
                if (i2 < 0) {
                    i2 += 100;
                }
                int max = Math.max(this.minLimit, Math.min(this.maxLimit, i2));
                BleProduct connBleProduct2 = BleManager.getInstance(MyApplication.getApplication()).getConnBleProduct(this.followerId);
                if (connBleProduct2 != null) {
                    connBleProduct2.writeSpeed(max);
                }
            }
            if (this.vibrateChannel == speedChannel) {
                int i3 = (int) (i * this.factorB);
                if (i3 < 0) {
                    i3 += 100;
                }
                int max2 = Math.max(this.minLimit, Math.min(this.maxLimit, i3));
                BleProduct connBleProduct3 = BleManager.getInstance(MyApplication.getApplication()).getConnBleProduct(this.followerId);
                if (connBleProduct3 != null) {
                    connBleProduct3.writeVibrate(max2);
                }
            }
        }

        public void applyRuleSwitch(boolean z, SpeedChannel speedChannel) {
            ProductModel productModel;
            if (!this.enabled || (productModel = this.mProductModel) == null) {
                LogUtil.e("applyRule enabled:" + this.enabled + ", mProductModel:" + this.mProductModel + ", followerId:" + this.followerId);
                return;
            }
            if (Constans.DEVICE_TYPE_FUN.equals(productModel.getDeviceType())) {
                LogUtil.e("applyRuleSwitch input:" + z + ", switchDelay:" + this.switchDelay);
                if (z && this.speedChannel == speedChannel) {
                    if (this.switchDelay > 0) {
                        getOrCreateHandler().sendEmptyMessageDelayed(3, this.switchDelay * 1000);
                        return;
                    }
                    BleProduct connBleProduct = BleManager.getInstance(MyApplication.getApplication()).getConnBleProduct(this.followerId);
                    if (connBleProduct != null) {
                        connBleProduct.writeOpenPiu();
                        return;
                    }
                    return;
                }
                return;
            }
            LogUtil.e("applyRuleSwitch followA:" + this.followA + ", followB:" + this.followB);
            LogUtil.e("applyRuleSwitch input:" + z + ", onValueA:" + this.onValueA + ", durationA:" + this.durationA);
            LogUtil.e("applyRuleSwitch input:" + z + ", onValueB:" + this.onValueB + ", durationB:" + this.durationB);
            if (this.followA && z && this.speedChannel == speedChannel) {
                BleProduct connBleProduct2 = BleManager.getInstance(MyApplication.getApplication()).getConnBleProduct(this.followerId);
                if (connBleProduct2 != null) {
                    connBleProduct2.writeSpeed(this.onValueA);
                }
                long currentTimeMillis = System.currentTimeMillis();
                long j = this.durationAEndTime;
                long j2 = (j > currentTimeMillis ? j - currentTimeMillis : 0L) + currentTimeMillis + (this.durationA * 1000);
                this.durationAEndTime = j2;
                long j3 = j2 - currentTimeMillis;
                getOrCreateHandler().removeMessages(1);
                LogUtil.e("applyRuleSwitch sendEmptyMessageDelayed:WHAT_DURATION_A delayMillis:" + j3);
                getOrCreateHandler().sendEmptyMessageDelayed(1, j3);
            }
            if (this.followB && z && this.vibrateChannel == speedChannel) {
                BleProduct connBleProduct3 = BleManager.getInstance(MyApplication.getApplication()).getConnBleProduct(this.followerId);
                if (connBleProduct3 != null) {
                    connBleProduct3.writeVibrate(this.onValueB);
                }
                long currentTimeMillis2 = System.currentTimeMillis();
                long j4 = this.durationBEndTime;
                long j5 = (j4 > currentTimeMillis2 ? j4 - currentTimeMillis2 : 0L) + currentTimeMillis2 + (this.durationB * 1000);
                this.durationBEndTime = j5;
                getOrCreateHandler().removeMessages(2);
                getOrCreateHandler().sendEmptyMessageDelayed(2, j5 - currentTimeMillis2);
            }
        }

        /* renamed from: clone, reason: merged with bridge method [inline-methods] */
        public ControlRule m606clone() throws CloneNotSupportedException {
            try {
                ControlRule controlRule = (ControlRule) super.clone();
                controlRule.durationAEndTime = 0L;
                controlRule.durationBEndTime = 0L;
                controlRule.mHandler = null;
                return controlRule;
            } catch (Exception e) {
                e.printStackTrace();
                return new ControlRule(this.followerId, true, this.mProductModel);
            }
        }

        public ConditionType getCondition() {
            return this.condition;
        }

        public int getDurationA() {
            return this.durationA;
        }

        public int getDurationB() {
            return this.durationB;
        }

        public float getFactorA() {
            return this.factorA;
        }

        public float getFactorB() {
            return this.factorB;
        }

        public int getMaxLimit() {
            return this.maxLimit;
        }

        public int getMinLimit() {
            return this.minLimit;
        }

        public int getOffValueA() {
            return this.offValueA;
        }

        public int getOffValueB() {
            return this.offValueB;
        }

        public int getOnValueA() {
            return this.onValueA;
        }

        public int getOnValueB() {
            return this.onValueB;
        }

        public SpeedChannel getSpeedChannel() {
            return this.speedChannel;
        }

        public int getSwitchDelay() {
            return this.switchDelay;
        }

        public int getThreshold() {
            return this.threshold;
        }

        public SpeedChannel getVibrateChannel() {
            return this.vibrateChannel;
        }

        public ProductModel getmProductModel() {
            return this.mProductModel;
        }

        public boolean isEnabled() {
            return this.enabled;
        }

        public boolean isFollowA() {
            return this.followA;
        }

        public boolean isFollowB() {
            return this.followB;
        }

        public boolean isTriggerState() {
            return this.triggerState;
        }

        public void setCondition(ConditionType conditionType) {
            this.condition = conditionType;
        }

        public void setDurationA(int i) {
            this.durationA = i;
        }

        public void setDurationB(int i) {
            this.durationB = i;
        }

        public void setEnabled(boolean z) {
            this.enabled = z;
        }

        public void setFactorA(float f) {
            this.factorA = Math.max(-2.0f, Math.min(2.0f, f));
        }

        public void setFactorB(float f) {
            this.factorB = Math.max(-2.0f, Math.min(2.0f, f));
        }

        public void setFollowA(boolean z) {
            this.followA = z;
        }

        public void setFollowB(boolean z) {
            this.followB = z;
        }

        public void setMaxLimit(int i) {
            this.maxLimit = i;
        }

        public void setMinLimit(int i) {
            this.minLimit = i;
        }

        public void setOffValueA(int i) {
            this.offValueA = i;
        }

        public void setOffValueB(int i) {
            this.offValueB = i;
        }

        public void setOnValueA(int i) {
            this.onValueA = i;
        }

        public void setOnValueB(int i) {
            this.onValueB = i;
        }

        public void setSpeedChannel(SpeedChannel speedChannel) {
            this.speedChannel = speedChannel;
        }

        public void setSwitchDelay(int i) {
            this.switchDelay = i;
        }

        public void setThreshold(int i) {
            this.threshold = i;
        }

        public void setTriggerState(boolean z) {
            this.triggerState = z;
        }

        public void setVibrateChannel(SpeedChannel speedChannel) {
            this.vibrateChannel = speedChannel;
        }

        public void setmProductModel(ProductModel productModel) {
            this.mProductModel = productModel;
        }

        public String toString() {
            StringBuilder sb = new StringBuilder("ControlRule{name='");
            ProductModel productModel = this.mProductModel;
            return sb.append(productModel != null ? productModel.getName() : "").append("'followerId='").append(this.followerId).append("', factorA=").append(this.factorA).append(", factorB=").append(this.factorB).append(", speedChannel=").append(this.speedChannel).append(", vibrateChannel=").append(this.vibrateChannel).append(", minLimit=").append(this.minLimit).append(", maxLimit=").append(this.maxLimit).append(", enabled=").append(this.enabled).append(", condition=").append(this.condition).append(", threshold=").append(this.threshold).append(", triggerState=").append(this.triggerState).append(", onValueA=").append(this.onValueA).append(", offValueA=").append(this.offValueA).append(", durationA=").append(this.durationA).append(", followA=").append(this.followA).append(", onValueB=").append(this.onValueB).append(", offValueB=").append(this.offValueB).append(", durationB=").append(this.durationB).append(", followB=").append(this.followB).append(", switchDelay=").append(this.switchDelay).append('}').toString();
        }
    }

    private ControlRuleManager() {
    }

    private void getDeviceRulesForLeaderId(String str) {
        String str2 = (String) SharedPreferencesUtil.get(MyApplication.getApplication(), "leader" + str);
        LogUtil.e("getRuleSharedPreferences name:leader" + str + ", json:" + str2);
        if (str2.isEmpty()) {
            this.deviceRules = new HashMap();
        } else {
            Map<String, ControlRule> map = (Map) this.gson.fromJson(str2, new TypeToken<Map<String, ControlRule>>() { // from class: com.hismith.hismithapp.manager.ControlRuleManager.1
            }.getType());
            if (map != null) {
                this.deviceRules = map;
                for (Map.Entry<String, ControlRule> entry : map.entrySet()) {
                    initValue(entry.getKey(), ProductUtil.getProductModel(MyApplication.getApplication(), entry.getKey()));
                    LogUtil.e("getDeviceRulesForLeaderId key:" + entry.getKey() + ", entry:" + entry.getValue().toString());
                }
            } else {
                this.deviceRules = new HashMap();
            }
        }
        for (Map.Entry<String, BleProduct> entry2 : BleManager.getInstance(MyApplication.getApplication()).getBleProducts().entrySet()) {
            if (!this.deviceRules.containsKey(entry2.getKey())) {
                setDeviceRule(entry2.getKey(), ProductUtil.getProductModel(MyApplication.getApplication(), entry2.getKey()));
            }
        }
    }

    public static synchronized ControlRuleManager getInstance() {
        ControlRuleManager controlRuleManager;
        synchronized (ControlRuleManager.class) {
            controlRuleManager = instance;
            if (controlRuleManager == null) {
                controlRuleManager = new ControlRuleManager();
                instance = controlRuleManager;
            }
        }
        return controlRuleManager;
    }

    private void initValue(String str, ProductModel productModel) {
        this.deviceRules.get(str).setMinLimit(0);
        this.deviceRules.get(str).setMaxLimit(100);
        this.deviceRules.get(str).setEnabled(true);
        this.deviceRules.get(str).setmProductModel(productModel);
    }

    public void applyRuleClosePiu(String str) {
        ControlRule deviceRule = getDeviceRule(str);
        if (deviceRule != null) {
            deviceRule.applyRuleSwitch(false, SpeedChannel.CHANNEL_A);
        }
    }

    public void applyRuleOpenPiu(String str) {
        LogUtil.e("applyRuleOpenPiu deviceId:" + str);
        ControlRule deviceRule = getDeviceRule(str);
        if (deviceRule != null) {
            deviceRule.applyRuleSwitch(true, SpeedChannel.CHANNEL_A);
        }
    }

    public void applyRuleToSpeed(String str, int i) {
        ControlRule deviceRule = getDeviceRule(str);
        if (deviceRule != null) {
            deviceRule.applyRule(i, SpeedChannel.CHANNEL_A);
        }
    }

    public void applyRuleToVibrate(String str, int i) {
        ControlRule deviceRule = getDeviceRule(str);
        if (deviceRule != null) {
            deviceRule.applyRule(i, SpeedChannel.CHANNEL_B);
        }
    }

    public void clearLeader() {
        this.leaderId = "";
        this.deviceRules = new HashMap();
    }

    public ControlRule getDeviceRule(String str) {
        if (!this.deviceRules.containsKey(str)) {
            this.deviceRules.put(str, new ControlRule(str, false));
        }
        return this.deviceRules.get(str);
    }

    public boolean isLeader(String str) {
        return !TextUtils.isEmpty(this.leaderId) && this.leaderId.equals(str);
    }

    public void removeDeviceRule(String str) {
        this.deviceRules.remove(str);
    }

    public void removeHandleCallback() {
        Iterator<Map.Entry<String, ControlRule>> it = this.deviceRules.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().getOrCreateHandler().removeCallbacksAndMessages(null);
        }
    }

    public void saveDeviceRulesForLeaderId() {
        Map<String, ControlRule> map;
        if (TextUtils.isEmpty(this.leaderId) || (map = this.deviceRules) == null || map.isEmpty()) {
            LogUtil.e("not saveRuleSharedPreferences leaderId:" + this.leaderId + ", deviceRules:" + this.deviceRules.toString());
            return;
        }
        String json = this.gson.toJson(this.deviceRules);
        LogUtil.e("saveRuleSharedPreferences name:leader" + this.leaderId + ", json:" + json);
        if (TextUtils.isEmpty(this.leaderId)) {
            return;
        }
        SharedPreferencesUtil.put(MyApplication.getApplication(), "leader" + this.leaderId, json);
    }

    public void setDeviceRule(String str, ControlRule controlRule) {
        this.deviceRules.put(str, controlRule);
    }

    public void setDeviceRule(String str, ProductModel productModel) {
        LogUtil.e("ControlRuleManager device:" + str + ", setDeviceRule:" + (productModel != null ? productModel.getName() : ""));
        if (TextUtils.isEmpty(this.leaderId)) {
            this.leaderId = str;
        }
        if (this.deviceRules.containsKey(str) && this.deviceRules.get(str).isEnabled()) {
            initValue(str, productModel);
            LogUtil.e("setDeviceRule " + productModel.getName() + " containsKey :" + this.deviceRules.get(str).toString());
            return;
        }
        if (productModel == null) {
            return;
        }
        if (!this.leaderId.equals(str)) {
            SpeedChannel speedChannel = SpeedChannel.CHANNEL_A;
            if (ProductUtil.hasFun(1)) {
                speedChannel = SpeedChannel.CHANNEL_B;
            }
            SpeedChannel speedChannel2 = speedChannel;
            if (Constans.DEVICE_TYPE_FUN.equals(productModel.getDeviceType())) {
                ControlRule controlRule = new ControlRule(str, ConditionType.EQUAL, 50, true, productModel);
                controlRule.setVibrateChannel(speedChannel2);
                this.deviceRules.put(str, controlRule);
            } else {
                ControlRule controlRule2 = new ControlRule(str, 1.0f, 1.0f, SpeedChannel.CHANNEL_A, speedChannel2, productModel);
                controlRule2.setOnValueA(50);
                controlRule2.setOffValueA(0);
                controlRule2.setFollowA(true);
                controlRule2.setOnValueB(50);
                controlRule2.setOffValueB(0);
                controlRule2.setFollowB(true);
                this.deviceRules.put(str, controlRule2);
            }
        } else if (Constans.DEVICE_TYPE_FUN.equals(productModel.getDeviceType())) {
            ControlRule controlRule3 = new ControlRule(str, true, productModel);
            LogUtil.e("setDeviceRule new ControlRule deviceId" + str + " ControlRule :" + controlRule3.toString());
            this.deviceRules.put(str, controlRule3);
        } else {
            this.deviceRules.put(str, new ControlRule(str, 1.0f, 1.0f, SpeedChannel.CHANNEL_A, SpeedChannel.CHANNEL_B, productModel));
        }
        for (Map.Entry<String, ControlRule> entry : this.deviceRules.entrySet()) {
            LogUtil.e("setDeviceRule deviceRules key:" + entry.getKey() + ", entry:" + entry.getValue().toString());
        }
    }

    public void setLeaderId(String str) {
        LogUtil.e("setLeaderId:" + str);
        this.leaderId = str;
        removeHandleCallback();
        getDeviceRulesForLeaderId(str);
    }
}
