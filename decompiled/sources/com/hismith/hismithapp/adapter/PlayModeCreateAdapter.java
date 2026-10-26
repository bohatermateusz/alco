package com.hismith.hismithapp.adapter;

import android.app.Activity;
import android.text.Editable;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.google.firebase.analytics.FirebaseAnalytics;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.dialog.CountPickerDialog;
import com.hismith.hismithapp.dialog.NumberPickerDialog;
import com.hismith.hismithapp.listener.SoftKeyBoardListener;
import com.hismith.hismithapp.model.PlayModeItemModel;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import java.util.List;
import java.util.regex.Pattern;

/* loaded from: classes3.dex */
public class PlayModeCreateAdapter extends RecyclerView.Adapter<BaseViewHolder> {
    public static final int ITEM_Cycle = 2;
    public static final int ITEM_Reversal = 3;
    public static final int ITEM_SMS = 4;
    public static final int ITEM_Single = 1;
    private Activity mContext;
    private List<PlayModeItemModel> mList;
    private OnDeleteEmptyListener mOnDeleteEmptyListener;
    private int mSerialId = 0;

    class BaseViewHolder extends RecyclerView.ViewHolder {
        private FrameLayout mLayoutDelete;

        public BaseViewHolder(View view) {
            super(view);
            this.mLayoutDelete = (FrameLayout) view.findViewById(R.id.item_mode_create_layout_delete);
        }
    }

    class CycleViewHolder extends BaseViewHolder {
        private TextView item_mode_create_txt_end_phase;
        private TextView item_mode_create_txt_start_phase;
        private ImageView mImgType;
        private LinearLayout mLayout;
        private LinearLayout mLayoutStart;
        private LinearLayout mLayoutTime;
        private TextView mTxtEnd;
        private TextView mTxtStart;
        private TextView mTxtTime;
        private TextView mTxtTimeValue;

        public CycleViewHolder(View view) {
            super(view);
            this.mLayout = (LinearLayout) view.findViewById(R.id.item_mode_layout);
            this.mLayoutStart = (LinearLayout) view.findViewById(R.id.item_mode_create_layout_start);
            this.mLayoutTime = (LinearLayout) view.findViewById(R.id.item_mode_create_layout_time);
            this.mImgType = (ImageView) view.findViewById(R.id.item_mode_create_img_type);
            this.mTxtStart = (TextView) view.findViewById(R.id.item_mode_create_txt_start);
            this.mTxtEnd = (TextView) view.findViewById(R.id.item_mode_create_txt_end);
            this.mTxtTime = (TextView) view.findViewById(R.id.item_mode_create_txt_time);
            this.mTxtTimeValue = (TextView) view.findViewById(R.id.item_mode_create_txt_time_value);
            this.item_mode_create_txt_start_phase = (TextView) view.findViewById(R.id.item_mode_create_txt_start_phase);
            this.item_mode_create_txt_end_phase = (TextView) view.findViewById(R.id.item_mode_create_txt_end_phase);
        }
    }

    public interface OnDeleteEmptyListener {
        void onEmpty();
    }

    class ReversalViewHolder extends BaseViewHolder {
        private ImageView mImgType;
        private LinearLayout mLayout;
        private TextView mTxtEnd;
        private TextView mTxtEndValue;
        private TextView mTxtSerial;

        public ReversalViewHolder(View view) {
            super(view);
            this.mLayout = (LinearLayout) view.findViewById(R.id.item_mode_layout);
            this.mImgType = (ImageView) view.findViewById(R.id.item_mode_create_img_type);
            this.mTxtSerial = (TextView) view.findViewById(R.id.item_mode_create_txt_serial);
            this.mTxtEnd = (TextView) view.findViewById(R.id.item_mode_create_txt_end);
            this.mTxtEndValue = (TextView) view.findViewById(R.id.item_mode_create_txt_end_value);
        }
    }

    class SMSViewHolder extends BaseViewHolder {
        private ImageView item_mode_create_img_type;
        private TextView item_mode_create_txt_end;
        private EditText item_mode_create_txt_end_position;
        private TextView item_mode_create_txt_end_speed;
        private EditText item_mode_create_txt_end_speed_value;
        private TextView item_mode_create_txt_serial;
        private TextView item_mode_create_txt_smooth;
        private TextView item_mode_create_txt_smooth_value;
        private TextView item_mode_create_txt_start;
        private EditText item_mode_create_txt_start_position;
        private TextView item_mode_create_txt_start_speed;
        private EditText item_mode_create_txt_start_speed_value;
        private TextView item_mode_create_txt_time;
        private TextView item_mode_create_txt_time_value;

        public SMSViewHolder(View view) {
            super(view);
            this.item_mode_create_img_type = (ImageView) view.findViewById(R.id.item_mode_create_img_type);
            this.item_mode_create_txt_serial = (TextView) view.findViewById(R.id.item_mode_create_txt_serial);
            this.item_mode_create_txt_start = (TextView) view.findViewById(R.id.item_mode_create_txt_start);
            this.item_mode_create_txt_end = (TextView) view.findViewById(R.id.item_mode_create_txt_end);
            this.item_mode_create_txt_smooth = (TextView) view.findViewById(R.id.item_mode_create_txt_smooth);
            this.item_mode_create_txt_start_speed = (TextView) view.findViewById(R.id.item_mode_create_txt_start_speed);
            this.item_mode_create_txt_end_speed = (TextView) view.findViewById(R.id.item_mode_create_txt_end_speed);
            this.item_mode_create_txt_time = (TextView) view.findViewById(R.id.item_mode_create_txt_time);
            this.item_mode_create_txt_start_position = (EditText) view.findViewById(R.id.item_mode_create_txt_start_position);
            this.item_mode_create_txt_end_position = (EditText) view.findViewById(R.id.item_mode_create_txt_end_position);
            this.item_mode_create_txt_start_speed_value = (EditText) view.findViewById(R.id.item_mode_create_txt_start_speed_value);
            this.item_mode_create_txt_end_speed_value = (EditText) view.findViewById(R.id.item_mode_create_txt_end_speed_value);
            this.item_mode_create_txt_smooth_value = (TextView) view.findViewById(R.id.item_mode_create_txt_smooth_value);
            this.item_mode_create_txt_time_value = (TextView) view.findViewById(R.id.item_mode_create_txt_time_value);
        }
    }

    class SingleViewHolder extends BaseViewHolder {
        private ImageView mImgType;
        private LinearLayout mLayout;
        private FrameLayout mLayoutConfirm;
        private LinearLayout mLayoutSpeed;
        private LinearLayout mLayoutStart;
        private LinearLayout mLayoutTime;
        private SeekBar mSeekSpeed;
        private TextView mTxtEnd;
        private EditText mTxtEndValue;
        private EditText mTxtEndValue_2;
        private TextView mTxtM1;
        private TextView mTxtM2;
        private TextView mTxtSerial;
        private TextView mTxtSpeed;
        private TextView mTxtSpeedLabel;
        private TextView mTxtStart;
        private EditText mTxtStartValue;
        private EditText mTxtStartValue_2;
        private TextView mTxtTime;
        private TextView mTxtTimeValue;

        public SingleViewHolder(View view) {
            super(view);
            this.mLayout = (LinearLayout) view.findViewById(R.id.item_mode_layout);
            this.mLayoutStart = (LinearLayout) view.findViewById(R.id.item_mode_create_layout_start);
            this.mLayoutTime = (LinearLayout) view.findViewById(R.id.item_mode_create_layout_time);
            this.mImgType = (ImageView) view.findViewById(R.id.item_mode_create_img_type);
            this.mTxtSerial = (TextView) view.findViewById(R.id.item_mode_create_txt_serial);
            this.mTxtStart = (TextView) view.findViewById(R.id.item_mode_create_txt_start);
            this.mTxtStartValue = (EditText) view.findViewById(R.id.item_mode_create_txt_start_value);
            this.mTxtEnd = (TextView) view.findViewById(R.id.item_mode_create_txt_end);
            this.mTxtEndValue = (EditText) view.findViewById(R.id.item_mode_create_txt_end_value);
            this.mTxtTime = (TextView) view.findViewById(R.id.item_mode_create_txt_time);
            this.mTxtTimeValue = (TextView) view.findViewById(R.id.item_mode_create_txt_time_value);
            this.mTxtM1 = (TextView) view.findViewById(R.id.item_mode_create_txt_m1);
            this.mTxtM2 = (TextView) view.findViewById(R.id.item_mode_create_txt_m2);
            this.mTxtStartValue_2 = (EditText) view.findViewById(R.id.item_mode_create_txt_start_value_2);
            this.mTxtEndValue_2 = (EditText) view.findViewById(R.id.item_mode_create_txt_end_value_2);
            this.mLayoutSpeed = (LinearLayout) view.findViewById(R.id.layout_speed);
            this.mTxtSpeedLabel = (TextView) view.findViewById(R.id.txt_speed_label);
            this.mTxtSpeed = (TextView) view.findViewById(R.id.txt_speed);
            this.mSeekSpeed = (SeekBar) view.findViewById(R.id.seek_speed);
            this.mLayoutConfirm = (FrameLayout) view.findViewById(R.id.layout_confirm);
            LanguageUtil.setText(PlayModeCreateAdapter.this.mContext, this.mTxtSpeedLabel, "speedTitle");
            this.mSeekSpeed.setMax(100);
        }
    }

    public PlayModeCreateAdapter(Activity activity, List<PlayModeItemModel> list, OnDeleteEmptyListener onDeleteEmptyListener) {
        this.mContext = activity;
        this.mList = list;
        this.mOnDeleteEmptyListener = onDeleteEmptyListener;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public boolean isNumeric(String str) {
        return Pattern.compile("[0-9]*").matcher(str).matches();
    }

    public void addCycle(PlayModeItemModel playModeItemModel) {
        this.mList.add(playModeItemModel);
    }

    public void addSingle(PlayModeItemModel playModeItemModel) {
        int i = this.mSerialId + 1;
        this.mSerialId = i;
        playModeItemModel.setSerialId(i);
        this.mList.add(playModeItemModel);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemViewType(int i) {
        if (this.mList.get(i).getStart() == 255) {
            return 3;
        }
        if ("1".equals(this.mList.get(i).getSingle()) || "2".equals(this.mList.get(i).getSingle())) {
            return 1;
        }
        if ("0".equals(this.mList.get(i).getSingle())) {
            return 2;
        }
        if ("S".equals(this.mList.get(i).getSingle())) {
            return 4;
        }
        return super.getItemViewType(i);
    }

    public List<PlayModeItemModel> getList() {
        return this.mList;
    }

    public int getmSerialId() {
        return this.mSerialId;
    }

    public void initData() {
        int i = 0;
        for (int i2 = 0; i2 < this.mList.size(); i2++) {
            boolean equals = "0".equals(this.mList.get(i2).getSingle());
            List<PlayModeItemModel> list = this.mList;
            if (equals) {
                if (list.get(i2).getStart() > i) {
                    this.mList.get(i2).setStart(i);
                }
                if (this.mList.get(i2).getEnd() > i) {
                    this.mList.get(i2).setEnd(i);
                }
            } else {
                i++;
                list.get(i2).setSerialId(i);
            }
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(final BaseViewHolder baseViewHolder, final int i) {
        LogUtil.e("data(" + i + ")=single:" + this.mList.get(i).getSingle() + ", start:" + this.mList.get(i).getStart() + ", end:" + this.mList.get(i).getEnd() + ", start2:" + this.mList.get(i).getStart2() + ", end2:" + this.mList.get(i).getEnd2() + ", time:" + this.mList.get(i).getSeconds() + "." + this.mList.get(i).getMilli());
        baseViewHolder.mLayoutDelete.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (i < PlayModeCreateAdapter.this.mList.size()) {
                    int i2 = i;
                    if (i2 == 0 && i2 + 1 < PlayModeCreateAdapter.this.mList.size() && "0".equals(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i + 1)).getSingle())) {
                        return;
                    }
                    PlayModeCreateAdapter.this.removeItem(i, true);
                }
            }
        });
        if (baseViewHolder instanceof SingleViewHolder) {
            final SingleViewHolder singleViewHolder = (SingleViewHolder) baseViewHolder;
            if (singleViewHolder.mTxtStartValue.getTag() instanceof TextWatcher) {
                singleViewHolder.mTxtStartValue.removeTextChangedListener((TextWatcher) singleViewHolder.mTxtStartValue.getTag());
            }
            if (singleViewHolder.mTxtEndValue.getTag() instanceof TextWatcher) {
                singleViewHolder.mTxtEndValue.removeTextChangedListener((TextWatcher) singleViewHolder.mTxtEndValue.getTag());
            }
            if (singleViewHolder.mTxtStartValue_2.getTag() instanceof TextWatcher) {
                singleViewHolder.mTxtStartValue_2.removeTextChangedListener((TextWatcher) singleViewHolder.mTxtStartValue_2.getTag());
            }
            if (singleViewHolder.mTxtEndValue_2.getTag() instanceof TextWatcher) {
                singleViewHolder.mTxtEndValue_2.removeTextChangedListener((TextWatcher) singleViewHolder.mTxtEndValue_2.getTag());
            }
            LanguageUtil.setText(this.mContext, singleViewHolder.mTxtStart, "startSpeed");
            LanguageUtil.setText(this.mContext, singleViewHolder.mTxtEnd, "endSpeed");
            LanguageUtil.setText(this.mContext, singleViewHolder.mTxtTime, "processTime");
            singleViewHolder.mLayoutStart.setVisibility(0);
            if (ProductUtil.isSMVType()) {
                singleViewHolder.mTxtM1.setVisibility(0);
                singleViewHolder.mTxtM2.setVisibility(0);
                singleViewHolder.mTxtStartValue_2.setVisibility(0);
                singleViewHolder.mTxtEndValue_2.setVisibility(0);
            } else {
                singleViewHolder.mTxtM1.setVisibility(8);
                singleViewHolder.mTxtM2.setVisibility(8);
                singleViewHolder.mTxtStartValue_2.setVisibility(8);
                singleViewHolder.mTxtEndValue_2.setVisibility(8);
                if (ProductUtil.isSMSType()) {
                    singleViewHolder.mLayoutStart.setVisibility(8);
                    LanguageUtil.setText(this.mContext, singleViewHolder.mTxtEnd, FirebaseAnalytics.Param.LOCATION);
                }
            }
            singleViewHolder.mLayout.setBackgroundResource(R.drawable.bg_mode_item_content);
            singleViewHolder.mTxtSerial.setText(LanguageUtil.getLanguageByKey(this.mContext, "No.") + "" + this.mList.get(i).getSerialId());
            singleViewHolder.mImgType.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_play_mode_item_single_2));
            singleViewHolder.mTxtStartValue.setText(this.mList.get(i).getStart() + "");
            singleViewHolder.mTxtEndValue.setText(this.mList.get(i).getEnd() + "");
            singleViewHolder.mTxtStartValue_2.setText(this.mList.get(i).getStart2() + "");
            singleViewHolder.mTxtEndValue_2.setText(this.mList.get(i).getEnd2() + "");
            TextWatcher textWatcher = new TextWatcher() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.2
                @Override // android.text.TextWatcher
                public void afterTextChanged(Editable editable) {
                }

                @Override // android.text.TextWatcher
                public void beforeTextChanged(CharSequence charSequence, int i2, int i3, int i4) {
                }

                @Override // android.text.TextWatcher
                public void onTextChanged(CharSequence charSequence, int i2, int i3, int i4) {
                    int i5;
                    LogUtil.e("start onTextChanged(" + i + "):" + charSequence.toString());
                    if ("1".equals(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getSingle()) || "2".equals(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getSingle())) {
                        if (TextUtils.isEmpty(singleViewHolder.mTxtStartValue.getText()) || !PlayModeCreateAdapter.this.isNumeric(singleViewHolder.mTxtStartValue.getText().toString())) {
                            i5 = 0;
                        } else {
                            i5 = Integer.parseInt(singleViewHolder.mTxtStartValue.getText().toString());
                            if (i5 > 100) {
                                singleViewHolder.mTxtStartValue.setText("100");
                                singleViewHolder.mTxtStartValue.setSelection(singleViewHolder.mTxtStartValue.getText().length());
                                i5 = 100;
                            }
                        }
                        ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).setStart(i5);
                    }
                }
            };
            singleViewHolder.mTxtStartValue.addTextChangedListener(textWatcher);
            singleViewHolder.mTxtStartValue.setTag(textWatcher);
            TextWatcher textWatcher2 = new TextWatcher() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.3
                @Override // android.text.TextWatcher
                public void afterTextChanged(Editable editable) {
                }

                @Override // android.text.TextWatcher
                public void beforeTextChanged(CharSequence charSequence, int i2, int i3, int i4) {
                }

                @Override // android.text.TextWatcher
                public void onTextChanged(CharSequence charSequence, int i2, int i3, int i4) {
                    int i5;
                    LogUtil.e("end onTextChanged(" + i + "):" + charSequence.toString());
                    if ("1".equals(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getSingle()) || "2".equals(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getSingle())) {
                        if (TextUtils.isEmpty(singleViewHolder.mTxtEndValue.getText()) || !PlayModeCreateAdapter.this.isNumeric(singleViewHolder.mTxtEndValue.getText().toString())) {
                            i5 = 0;
                        } else {
                            i5 = Integer.parseInt(singleViewHolder.mTxtEndValue.getText().toString());
                            if (i5 > 100) {
                                singleViewHolder.mTxtEndValue.setText("100");
                                singleViewHolder.mTxtEndValue.setSelection(singleViewHolder.mTxtEndValue.getText().length());
                                i5 = 100;
                            }
                        }
                        ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).setEnd(i5);
                    }
                }
            };
            singleViewHolder.mTxtEndValue.addTextChangedListener(textWatcher2);
            singleViewHolder.mTxtEndValue.setTag(textWatcher2);
            TextWatcher textWatcher3 = new TextWatcher() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.4
                @Override // android.text.TextWatcher
                public void afterTextChanged(Editable editable) {
                }

                @Override // android.text.TextWatcher
                public void beforeTextChanged(CharSequence charSequence, int i2, int i3, int i4) {
                }

                @Override // android.text.TextWatcher
                public void onTextChanged(CharSequence charSequence, int i2, int i3, int i4) {
                    int i5;
                    if ("1".equals(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getSingle()) || "2".equals(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getSingle())) {
                        if (TextUtils.isEmpty(singleViewHolder.mTxtStartValue_2.getText()) || !PlayModeCreateAdapter.this.isNumeric(singleViewHolder.mTxtStartValue_2.getText().toString())) {
                            i5 = 0;
                        } else {
                            i5 = Integer.parseInt(singleViewHolder.mTxtStartValue_2.getText().toString());
                            if (i5 > 100) {
                                singleViewHolder.mTxtStartValue_2.setText("100");
                                singleViewHolder.mTxtStartValue_2.setSelection(singleViewHolder.mTxtStartValue_2.getText().length());
                                i5 = 100;
                            }
                        }
                        ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).setStart2(i5);
                    }
                }
            };
            singleViewHolder.mTxtStartValue_2.addTextChangedListener(textWatcher3);
            singleViewHolder.mTxtStartValue_2.setTag(textWatcher3);
            TextWatcher textWatcher4 = new TextWatcher() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.5
                @Override // android.text.TextWatcher
                public void afterTextChanged(Editable editable) {
                }

                @Override // android.text.TextWatcher
                public void beforeTextChanged(CharSequence charSequence, int i2, int i3, int i4) {
                }

                @Override // android.text.TextWatcher
                public void onTextChanged(CharSequence charSequence, int i2, int i3, int i4) {
                    int i5;
                    if ("1".equals(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getSingle()) || "2".equals(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getSingle())) {
                        if (TextUtils.isEmpty(singleViewHolder.mTxtEndValue_2.getText()) || !PlayModeCreateAdapter.this.isNumeric(singleViewHolder.mTxtEndValue_2.getText().toString())) {
                            i5 = 0;
                        } else {
                            i5 = Integer.parseInt(singleViewHolder.mTxtEndValue_2.getText().toString());
                            if (i5 > 100) {
                                singleViewHolder.mTxtEndValue_2.setText("100");
                                singleViewHolder.mTxtEndValue_2.setSelection(singleViewHolder.mTxtEndValue_2.getText().length());
                                i5 = 100;
                            }
                        }
                        ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).setEnd2(i5);
                    }
                }
            };
            singleViewHolder.mTxtEndValue_2.addTextChangedListener(textWatcher4);
            singleViewHolder.mTxtEndValue_2.setTag(textWatcher4);
            singleViewHolder.mTxtTimeValue.setText(this.mList.get(i).getSeconds() + "." + this.mList.get(i).getMilli());
            singleViewHolder.mTxtTimeValue.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.6
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    new NumberPickerDialog(PlayModeCreateAdapter.this.mContext, ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getSeconds(), ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getMilli()) { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.6.1
                        @Override // com.hismith.hismithapp.dialog.NumberPickerDialog
                        public void btnSubmit(int i2, int i3) {
                            singleViewHolder.mTxtTimeValue.setText(i2 + "." + i3);
                            ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).setSeconds(i2);
                            ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).setMilli(i3);
                        }
                    }.show();
                }
            });
        } else if (baseViewHolder instanceof CycleViewHolder) {
            final CycleViewHolder cycleViewHolder = (CycleViewHolder) baseViewHolder;
            cycleViewHolder.item_mode_create_txt_start_phase.setText(this.mList.get(i).getStart() + "");
            cycleViewHolder.item_mode_create_txt_end_phase.setText(this.mList.get(i).getEnd() + "");
            LanguageUtil.setText(this.mContext, cycleViewHolder.mTxtStart, "startPhase");
            LanguageUtil.setText(this.mContext, cycleViewHolder.mTxtEnd, "endPhase");
            LanguageUtil.setText(this.mContext, cycleViewHolder.mTxtTime, "cycles");
            cycleViewHolder.mImgType.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_play_mode_item_loop_2));
            if (this.mList.get(i).getCount() != 0) {
                cycleViewHolder.mTxtTimeValue.setText(this.mList.get(i).getCount() + "");
            } else {
                cycleViewHolder.mTxtTimeValue.setText("");
            }
            cycleViewHolder.mTxtTimeValue.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.7
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    new CountPickerDialog(PlayModeCreateAdapter.this.mContext, Integer.valueOf(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getCount()), 1, 100) { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.7.1
                        @Override // com.hismith.hismithapp.dialog.CountPickerDialog
                        public void btnSubmit(int i2) {
                            cycleViewHolder.mTxtTimeValue.setText(i2 + "");
                            ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).setCount(i2);
                        }
                    }.show();
                }
            });
            cycleViewHolder.item_mode_create_txt_start_phase.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.8
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    int i2 = 0;
                    for (int i3 = 0; i3 < i; i3++) {
                        if (!"0".equals(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i3)).getSingle())) {
                            i2 = ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i3)).getSerialId();
                        }
                    }
                    Activity activity = PlayModeCreateAdapter.this.mContext;
                    Integer valueOf = Integer.valueOf(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getStart());
                    int i4 = 1;
                    if (((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getEnd() != 0) {
                        i2 = ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getEnd();
                    }
                    new CountPickerDialog(activity, valueOf, i4, Integer.valueOf(i2)) { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.8.1
                        @Override // com.hismith.hismithapp.dialog.CountPickerDialog
                        public void btnSubmit(int i5) {
                            cycleViewHolder.item_mode_create_txt_start_phase.setText(i5 + "");
                            ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).setStart(i5);
                        }
                    }.show();
                }
            });
            cycleViewHolder.item_mode_create_txt_end_phase.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.9
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    int i2 = 0;
                    for (int i3 = 0; i3 < i; i3++) {
                        if (!"0".equals(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i3)).getSingle())) {
                            i2 = ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i3)).getSerialId();
                        }
                    }
                    new CountPickerDialog(PlayModeCreateAdapter.this.mContext, Integer.valueOf(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getEnd()), Integer.valueOf(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getStart() != 0 ? ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getStart() : 1), Integer.valueOf(i2)) { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.9.1
                        @Override // com.hismith.hismithapp.dialog.CountPickerDialog
                        public void btnSubmit(int i4) {
                            cycleViewHolder.item_mode_create_txt_end_phase.setText(i4 + "");
                            ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).setEnd(i4);
                        }
                    }.show();
                }
            });
        } else if (baseViewHolder instanceof ReversalViewHolder) {
            final ReversalViewHolder reversalViewHolder = (ReversalViewHolder) baseViewHolder;
            LanguageUtil.setText(this.mContext, reversalViewHolder.mTxtEnd, "rotate");
            reversalViewHolder.mTxtSerial.setText(LanguageUtil.getLanguageByKey(this.mContext, "No.") + "" + this.mList.get(i).getSerialId());
            reversalViewHolder.mImgType.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_play_mode_item_reversal_2));
            final String[] strArr = {LanguageUtil.getLanguageByKey(this.mContext, "forward"), LanguageUtil.getLanguageByKey(this.mContext, "reverse")};
            if (this.mList.get(i).getEnd() > 2) {
                this.mList.get(i).setEnd(0);
            }
            reversalViewHolder.mTxtEndValue.setText(strArr[this.mList.get(i).getEnd()]);
            reversalViewHolder.mTxtEndValue.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.10
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    new CountPickerDialog(PlayModeCreateAdapter.this.mContext, Integer.valueOf(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getEnd()), 0, 1, strArr) { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.10.1
                        @Override // com.hismith.hismithapp.dialog.CountPickerDialog
                        public void btnSubmit(int i2) {
                            reversalViewHolder.mTxtEndValue.setText(strArr[i2]);
                            ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).setEnd(i2);
                        }
                    }.show();
                }
            });
        } else if (baseViewHolder instanceof SMSViewHolder) {
            final SMSViewHolder sMSViewHolder = (SMSViewHolder) baseViewHolder;
            if (sMSViewHolder.item_mode_create_txt_start_position.getTag() instanceof TextWatcher) {
                sMSViewHolder.item_mode_create_txt_start_position.removeTextChangedListener((TextWatcher) sMSViewHolder.item_mode_create_txt_start_position.getTag());
            }
            if (sMSViewHolder.item_mode_create_txt_end_position.getTag() instanceof TextWatcher) {
                sMSViewHolder.item_mode_create_txt_end_position.removeTextChangedListener((TextWatcher) sMSViewHolder.item_mode_create_txt_end_position.getTag());
            }
            if (sMSViewHolder.item_mode_create_txt_start_speed_value.getTag() instanceof TextWatcher) {
                sMSViewHolder.item_mode_create_txt_start_speed_value.removeTextChangedListener((TextWatcher) sMSViewHolder.item_mode_create_txt_start_speed_value.getTag());
            }
            if (sMSViewHolder.item_mode_create_txt_end_speed_value.getTag() instanceof TextWatcher) {
                sMSViewHolder.item_mode_create_txt_end_speed_value.removeTextChangedListener((TextWatcher) sMSViewHolder.item_mode_create_txt_end_speed_value.getTag());
            }
            sMSViewHolder.item_mode_create_txt_start_position.setOnFocusChangeListener(null);
            sMSViewHolder.item_mode_create_txt_end_position.setOnFocusChangeListener(null);
            LanguageUtil.setText(this.mContext, sMSViewHolder.item_mode_create_txt_start, "maxStroke");
            LanguageUtil.setText(this.mContext, sMSViewHolder.item_mode_create_txt_end, "Depth");
            LanguageUtil.setText(this.mContext, sMSViewHolder.item_mode_create_txt_smooth, "smoothness");
            LanguageUtil.setText(this.mContext, sMSViewHolder.item_mode_create_txt_start_speed, "startSpeed");
            LanguageUtil.setText(this.mContext, sMSViewHolder.item_mode_create_txt_end_speed, "endSpeed");
            LanguageUtil.setText(this.mContext, sMSViewHolder.item_mode_create_txt_time, "processTime");
            if (this.mList.get(i).getEnd() - this.mList.get(i).getStart() < 10) {
                this.mList.get(i).setEnd(this.mList.get(i).getStart() + 10);
            }
            sMSViewHolder.item_mode_create_txt_serial.setText(LanguageUtil.getLanguageByKey(this.mContext, "No.") + "" + this.mList.get(i).getSerialId());
            sMSViewHolder.item_mode_create_img_type.setImageResource(R.mipmap.ic_play_mode_item_single_2);
            sMSViewHolder.item_mode_create_txt_start_position.setText(this.mList.get(i).getStart() + "");
            sMSViewHolder.item_mode_create_txt_end_position.setText(this.mList.get(i).getEnd() + "");
            sMSViewHolder.item_mode_create_txt_start_speed_value.setText(this.mList.get(i).getStart2() + "");
            sMSViewHolder.item_mode_create_txt_end_speed_value.setText(this.mList.get(i).getEnd2() + "");
            sMSViewHolder.item_mode_create_txt_smooth_value.setText(this.mList.get(i).getSmooth() + "");
            sMSViewHolder.item_mode_create_txt_time_value.setText(this.mList.get(i).getSeconds() + "." + this.mList.get(i).getMilli());
            LogUtil.e(String.format("数据 model: start:%d, end:%d, start2:%d, end2:%d", Integer.valueOf(this.mList.get(i).getStart()), Integer.valueOf(this.mList.get(i).getEnd()), Integer.valueOf(this.mList.get(i).getStart2()), Integer.valueOf(this.mList.get(i).getEnd2())));
            TextWatcher textWatcher5 = new TextWatcher() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.11
                @Override // android.text.TextWatcher
                public void afterTextChanged(Editable editable) {
                }

                @Override // android.text.TextWatcher
                public void beforeTextChanged(CharSequence charSequence, int i2, int i3, int i4) {
                }

                @Override // android.text.TextWatcher
                public void onTextChanged(CharSequence charSequence, int i2, int i3, int i4) {
                    LogUtil.e("start onTextChanged(" + i + "):" + charSequence.toString());
                    if ("S".equals(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getSingle())) {
                        if (!TextUtils.isEmpty(sMSViewHolder.item_mode_create_txt_start_position.getText()) && PlayModeCreateAdapter.this.isNumeric(sMSViewHolder.item_mode_create_txt_start_position.getText().toString())) {
                            String obj = sMSViewHolder.item_mode_create_txt_start_position.getText().toString();
                            int parseInt = Integer.parseInt(obj);
                            int end = ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getEnd() - 10;
                            r3 = end >= 0 ? end : 0;
                            if (parseInt > r3) {
                                sMSViewHolder.item_mode_create_txt_start_position.setText(r3 + "");
                                sMSViewHolder.item_mode_create_txt_start_position.setSelection(sMSViewHolder.item_mode_create_txt_start_position.getText().length());
                            } else {
                                if (obj.length() > 3) {
                                    sMSViewHolder.item_mode_create_txt_start_position.setText(parseInt + "");
                                    sMSViewHolder.item_mode_create_txt_start_position.setSelection(sMSViewHolder.item_mode_create_txt_start_position.getText().length());
                                }
                                r3 = parseInt;
                            }
                        }
                        ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).setStart(r3);
                    }
                }
            };
            sMSViewHolder.item_mode_create_txt_start_position.addTextChangedListener(textWatcher5);
            sMSViewHolder.item_mode_create_txt_start_position.setTag(textWatcher5);
            sMSViewHolder.item_mode_create_txt_start_position.setOnFocusChangeListener(new View.OnFocusChangeListener() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.12
                @Override // android.view.View.OnFocusChangeListener
                public void onFocusChange(View view, boolean z) {
                }
            });
            TextWatcher textWatcher6 = new TextWatcher() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.13
                @Override // android.text.TextWatcher
                public void afterTextChanged(Editable editable) {
                }

                @Override // android.text.TextWatcher
                public void beforeTextChanged(CharSequence charSequence, int i2, int i3, int i4) {
                }

                @Override // android.text.TextWatcher
                public void onTextChanged(CharSequence charSequence, int i2, int i3, int i4) {
                    int i5;
                    LogUtil.e("end onTextChanged(" + i + "):" + charSequence.toString());
                    if ("S".equals(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getSingle())) {
                        if (TextUtils.isEmpty(sMSViewHolder.item_mode_create_txt_end_position.getText()) || !PlayModeCreateAdapter.this.isNumeric(sMSViewHolder.item_mode_create_txt_end_position.getText().toString())) {
                            i5 = 0;
                        } else {
                            String obj = sMSViewHolder.item_mode_create_txt_end_position.getText().toString();
                            i5 = Integer.parseInt(obj);
                            if (i5 > 100) {
                                sMSViewHolder.item_mode_create_txt_end_position.setText("100");
                                sMSViewHolder.item_mode_create_txt_end_position.setSelection(sMSViewHolder.item_mode_create_txt_end_position.getText().length());
                                i5 = 100;
                            } else if (obj.length() > 3) {
                                sMSViewHolder.item_mode_create_txt_end_position.setText(i5 + "");
                                sMSViewHolder.item_mode_create_txt_end_position.setSelection(sMSViewHolder.item_mode_create_txt_end_position.getText().length());
                            }
                        }
                        ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).setEnd(i5);
                    }
                }
            };
            sMSViewHolder.item_mode_create_txt_end_position.addTextChangedListener(textWatcher6);
            sMSViewHolder.item_mode_create_txt_end_position.setTag(textWatcher6);
            sMSViewHolder.item_mode_create_txt_end_position.setOnFocusChangeListener(new View.OnFocusChangeListener() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.14
                @Override // android.view.View.OnFocusChangeListener
                public void onFocusChange(View view, boolean z) {
                    int end = ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getEnd();
                    int start = ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getStart() + 10;
                    if (end < start) {
                        sMSViewHolder.item_mode_create_txt_end_position.setText(start + "");
                        ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).setEnd(start);
                    }
                }
            });
            TextWatcher textWatcher7 = new TextWatcher() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.15
                @Override // android.text.TextWatcher
                public void afterTextChanged(Editable editable) {
                }

                @Override // android.text.TextWatcher
                public void beforeTextChanged(CharSequence charSequence, int i2, int i3, int i4) {
                }

                @Override // android.text.TextWatcher
                public void onTextChanged(CharSequence charSequence, int i2, int i3, int i4) {
                    int i5;
                    LogUtil.e("speed onTextChanged(" + i + "):" + charSequence.toString());
                    if ("S".equals(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getSingle())) {
                        if (TextUtils.isEmpty(sMSViewHolder.item_mode_create_txt_start_speed_value.getText()) || !PlayModeCreateAdapter.this.isNumeric(sMSViewHolder.item_mode_create_txt_start_speed_value.getText().toString())) {
                            i5 = 0;
                        } else {
                            String obj = sMSViewHolder.item_mode_create_txt_start_speed_value.getText().toString();
                            i5 = Integer.parseInt(obj);
                            if (i5 > 100) {
                                sMSViewHolder.item_mode_create_txt_start_speed_value.setText("100");
                                sMSViewHolder.item_mode_create_txt_start_speed_value.setSelection(sMSViewHolder.item_mode_create_txt_start_speed_value.getText().length());
                                i5 = 100;
                            } else if (obj.length() > 3) {
                                sMSViewHolder.item_mode_create_txt_start_speed_value.setText(i5 + "");
                                sMSViewHolder.item_mode_create_txt_start_speed_value.setSelection(sMSViewHolder.item_mode_create_txt_start_speed_value.getText().length());
                            }
                        }
                        ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).setStart2(i5);
                    }
                }
            };
            sMSViewHolder.item_mode_create_txt_start_speed_value.addTextChangedListener(textWatcher7);
            sMSViewHolder.item_mode_create_txt_start_speed_value.setTag(textWatcher7);
            TextWatcher textWatcher8 = new TextWatcher() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.16
                @Override // android.text.TextWatcher
                public void afterTextChanged(Editable editable) {
                }

                @Override // android.text.TextWatcher
                public void beforeTextChanged(CharSequence charSequence, int i2, int i3, int i4) {
                }

                @Override // android.text.TextWatcher
                public void onTextChanged(CharSequence charSequence, int i2, int i3, int i4) {
                    int i5;
                    LogUtil.e("speed onTextChanged(" + i + "):" + charSequence.toString());
                    if ("S".equals(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getSingle())) {
                        if (TextUtils.isEmpty(sMSViewHolder.item_mode_create_txt_end_speed_value.getText()) || !PlayModeCreateAdapter.this.isNumeric(sMSViewHolder.item_mode_create_txt_end_speed_value.getText().toString())) {
                            i5 = 0;
                        } else {
                            String obj = sMSViewHolder.item_mode_create_txt_end_speed_value.getText().toString();
                            i5 = Integer.parseInt(obj);
                            if (i5 > 100) {
                                sMSViewHolder.item_mode_create_txt_end_speed_value.setText("100");
                                sMSViewHolder.item_mode_create_txt_end_speed_value.setSelection(sMSViewHolder.item_mode_create_txt_end_speed_value.getText().length());
                                i5 = 100;
                            } else if (obj.length() > 3) {
                                sMSViewHolder.item_mode_create_txt_end_speed_value.setText(i5 + "");
                                sMSViewHolder.item_mode_create_txt_end_speed_value.setSelection(sMSViewHolder.item_mode_create_txt_end_speed_value.getText().length());
                            }
                        }
                        ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).setEnd2(i5);
                    }
                }
            };
            sMSViewHolder.item_mode_create_txt_end_speed_value.addTextChangedListener(textWatcher8);
            sMSViewHolder.item_mode_create_txt_end_speed_value.setTag(textWatcher8);
            sMSViewHolder.item_mode_create_txt_smooth_value.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.17
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    new CountPickerDialog(PlayModeCreateAdapter.this.mContext, Integer.valueOf(((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getSmooth()), 0, 10) { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.17.1
                        @Override // com.hismith.hismithapp.dialog.CountPickerDialog
                        public void btnSubmit(int i2) {
                            sMSViewHolder.item_mode_create_txt_smooth_value.setText(i2 + "");
                            ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).setSmooth(i2);
                        }
                    }.show();
                }
            });
            sMSViewHolder.item_mode_create_txt_time_value.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.18
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    new NumberPickerDialog(PlayModeCreateAdapter.this.mContext, ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getSeconds(), ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).getMilli()) { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.18.1
                        @Override // com.hismith.hismithapp.dialog.NumberPickerDialog
                        public void btnSubmit(int i2, int i3) {
                            sMSViewHolder.item_mode_create_txt_time_value.setText(i2 + "." + i3);
                            ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).setSeconds(i2);
                            ((PlayModeItemModel) PlayModeCreateAdapter.this.mList.get(i)).setMilli(i3);
                        }
                    }.show();
                }
            });
        }
        new SoftKeyBoardListener(this.mContext).setListener(new SoftKeyBoardListener.OnSoftKeyBoardChangeListener() { // from class: com.hismith.hismithapp.adapter.PlayModeCreateAdapter.19
            @Override // com.hismith.hismithapp.listener.SoftKeyBoardListener.OnSoftKeyBoardChangeListener
            public void keyBoardHide(int i2) {
                baseViewHolder.itemView.setFocusable(true);
                baseViewHolder.itemView.setFocusableInTouchMode(true);
                baseViewHolder.itemView.requestFocus();
            }

            @Override // com.hismith.hismithapp.listener.SoftKeyBoardListener.OnSoftKeyBoardChangeListener
            public void keyBoardShow(int i2) {
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public BaseViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        if (i == 1) {
            return new SingleViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_play_mode_create_single, viewGroup, false));
        }
        if (i == 2) {
            return new CycleViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_play_mode_create_cycle, viewGroup, false));
        }
        if (i == 3) {
            return new ReversalViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_play_mode_create_reversal, viewGroup, false));
        }
        if (i != 4) {
            return null;
        }
        return new SMSViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_play_mode_create_sms, viewGroup, false));
    }

    public void removeItem(int i, boolean z) {
        OnDeleteEmptyListener onDeleteEmptyListener;
        if (!"0".equals(this.mList.get(i).getSingle())) {
            this.mSerialId--;
        }
        this.mList.remove(i);
        notifyItemRemoved(i);
        notifyItemRangeChanged(i, this.mList.size());
        if (z) {
            initData();
            notifyDataSetChanged();
        }
        if (this.mList.size() != 0 || (onDeleteEmptyListener = this.mOnDeleteEmptyListener) == null) {
            return;
        }
        onDeleteEmptyListener.onEmpty();
    }
}
