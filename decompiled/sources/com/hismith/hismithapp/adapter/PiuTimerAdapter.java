package com.hismith.hismithapp.adapter;

import android.app.Activity;
import android.text.Editable;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import com.google.gson.Gson;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.listener.SoftKeyBoardListener;
import com.hismith.hismithapp.manager.PiuTimerManager;
import com.hismith.hismithapp.model.PiuTimerModel;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import java.util.List;

/* loaded from: classes3.dex */
public class PiuTimerAdapter extends RecyclerView.Adapter<PiuTimerViewHolder> {
    private Gson gson = new Gson();
    private Activity mActivity;
    private List<PiuTimerModel> mList;
    private int showPiuTimerMode;

    class PiuTimerViewHolder extends RecyclerView.ViewHolder {
        private EditText edit_time;
        private ImageView img_delete;
        private ImageView img_play_bg;
        private ConstraintLayout layout_content;
        private ConstraintLayout layout_edit;
        private ConstraintLayout layout_item;
        private ConstraintLayout layout_title;
        private SoftKeyBoardListener listener;
        private TextView txt_content_title;
        private TextView txt_title;

        public PiuTimerViewHolder(View view) {
            super(view);
            this.img_play_bg = (ImageView) view.findViewById(R.id.img_play_bg);
            this.layout_item = (ConstraintLayout) view.findViewById(R.id.layout_item);
            this.layout_title = (ConstraintLayout) view.findViewById(R.id.layout_title);
            this.layout_content = (ConstraintLayout) view.findViewById(R.id.layout_content);
            this.txt_title = (TextView) view.findViewById(R.id.txt_title);
            this.txt_content_title = (TextView) view.findViewById(R.id.txt_content_title);
            this.img_delete = (ImageView) view.findViewById(R.id.img_delete);
            this.layout_edit = (ConstraintLayout) view.findViewById(R.id.layout_edit);
            this.edit_time = (EditText) view.findViewById(R.id.edit_time);
            LanguageUtil.setText(PiuTimerAdapter.this.mActivity, this.txt_title, "nextSprayTime");
            SoftKeyBoardListener softKeyBoardListener = new SoftKeyBoardListener(PiuTimerAdapter.this.mActivity);
            this.listener = softKeyBoardListener;
            softKeyBoardListener.setListener(new SoftKeyBoardListener.OnSoftKeyBoardChangeListener() { // from class: com.hismith.hismithapp.adapter.PiuTimerAdapter.PiuTimerViewHolder.1
                @Override // com.hismith.hismithapp.listener.SoftKeyBoardListener.OnSoftKeyBoardChangeListener
                public void keyBoardHide(int i) {
                    String obj = PiuTimerViewHolder.this.edit_time.getText().toString();
                    if (!TextUtils.isEmpty(obj) && obj.startsWith("0") && obj.length() > 1) {
                        PiuTimerViewHolder.this.edit_time.setText(Integer.parseInt(obj) + "");
                    }
                    PiuTimerViewHolder.this.layout_item.setFocusable(true);
                    PiuTimerViewHolder.this.layout_item.setFocusableInTouchMode(true);
                    PiuTimerViewHolder.this.layout_item.requestFocus();
                }

                @Override // com.hismith.hismithapp.listener.SoftKeyBoardListener.OnSoftKeyBoardChangeListener
                public void keyBoardShow(int i) {
                }
            });
        }
    }

    class TextChangeSaveShared implements TextWatcher {
        private EditText editText;
        private int position;

        public TextChangeSaveShared(EditText editText, int i) {
            this.editText = editText;
            this.position = i;
        }

        @Override // android.text.TextWatcher
        public void afterTextChanged(Editable editable) {
        }

        @Override // android.text.TextWatcher
        public void beforeTextChanged(CharSequence charSequence, int i, int i2, int i3) {
        }

        @Override // android.text.TextWatcher
        public void onTextChanged(CharSequence charSequence, int i, int i2, int i3) {
            LogUtil.e("textsssd onTextChanged " + ((Object) charSequence) + ", start:" + i + ", before:" + i2 + ", count:" + i3 + ", position:" + this.position);
            try {
                String obj = charSequence.toString();
                if (obj.isEmpty()) {
                    return;
                }
                int parseInt = Integer.parseInt(obj);
                if (obj.length() > 3) {
                    if (parseInt > 999) {
                        parseInt = 999;
                    }
                    this.editText.setText(parseInt + "");
                    EditText editText = this.editText;
                    editText.setSelection(editText.length());
                }
            } catch (NumberFormatException e) {
                e.printStackTrace();
            }
        }
    }

    public PiuTimerAdapter(Activity activity, int i, List<PiuTimerModel> list) {
        this.mActivity = activity;
        this.showPiuTimerMode = i;
        this.mList = list;
    }

    public void addItem() {
        int i;
        if (this.mList.isEmpty()) {
            i = 0;
        } else {
            i = this.mList.get(r0.size() - 1).getSecond();
        }
        this.mList.add(new PiuTimerModel(i));
        PiuTimerManager.getInstance(this.mActivity).saveShared(this.mList, this.showPiuTimerMode);
        notifyItemInserted(this.mList.size() - 1);
        notifyItemRangeChanged(this.mList.size() - 1, this.mList.size());
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        List<PiuTimerModel> list = this.mList;
        if (list != null) {
            return list.size();
        }
        return 0;
    }

    public List<PiuTimerModel> getmList() {
        return this.mList;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public /* bridge */ /* synthetic */ void onBindViewHolder(PiuTimerViewHolder piuTimerViewHolder, int i, List list) {
        onBindViewHolder2(piuTimerViewHolder, i, (List<Object>) list);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(final PiuTimerViewHolder piuTimerViewHolder, final int i) {
        LogUtil.e("piuTimer广播 adapter 显示list：" + this.mList.toString());
        if (piuTimerViewHolder.edit_time.getTag() instanceof TextWatcher) {
            piuTimerViewHolder.edit_time.removeTextChangedListener((TextWatcher) piuTimerViewHolder.edit_time.getTag());
        }
        if (PiuTimerManager.getInstance(this.mActivity).getmPlayLocalMode() != this.showPiuTimerMode || i >= PiuTimerManager.getInstance(this.mActivity).getPlayPosition()) {
            piuTimerViewHolder.layout_title.setBackgroundResource(R.drawable.bg_item_piu_timer_title);
            piuTimerViewHolder.img_delete.setImageResource(R.mipmap.ic_play_mode_item_delete3);
            piuTimerViewHolder.layout_edit.setBackgroundResource(R.drawable.bg_item_piu_timer_edit);
            piuTimerViewHolder.txt_content_title.setTextColor(this.mActivity.getColor(R.color.piu_adapter_content_text_color));
            piuTimerViewHolder.edit_time.setTextColor(this.mActivity.getColor(R.color.piu_adapter_edit_text_color));
        } else {
            piuTimerViewHolder.layout_title.setBackgroundResource(R.drawable.bg_item_piu_timer_title_gray);
            piuTimerViewHolder.img_delete.setImageResource(R.mipmap.ic_play_mode_item_delete4);
            piuTimerViewHolder.layout_edit.setBackground(null);
            piuTimerViewHolder.txt_content_title.setTextColor(this.mActivity.getColor(R.color.piu_adapter_edit_text_gray_color));
            piuTimerViewHolder.edit_time.setTextColor(this.mActivity.getColor(R.color.piu_adapter_edit_text_gray_color));
        }
        if (PiuTimerManager.getInstance(this.mActivity).getmPlayLocalMode() == this.showPiuTimerMode && PiuTimerManager.getInstance(this.mActivity).getPlayPosition() == i) {
            LogUtil.e("piuTimer广播 adapter 显示：" + i);
            if (PiuTimerManager.getInstance(this.mActivity).isPlay()) {
                piuTimerViewHolder.edit_time.setEnabled(false);
                piuTimerViewHolder.edit_time.setText(PiuTimerManager.getInstance(this.mActivity).getmProgress() + "");
                piuTimerViewHolder.edit_time.setTextColor(this.mActivity.getColor(R.color.item_piu_playing_color));
            } else {
                piuTimerViewHolder.edit_time.setEnabled(true);
                piuTimerViewHolder.edit_time.setText(this.mList.get(i).getSecond() + "");
            }
            piuTimerViewHolder.img_play_bg.setVisibility(0);
        } else {
            piuTimerViewHolder.img_play_bg.setVisibility(4);
            piuTimerViewHolder.edit_time.setEnabled(true);
            piuTimerViewHolder.edit_time.setText(this.mList.get(i).getSecond() + "");
        }
        piuTimerViewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.PiuTimerAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                PiuTimerManager.getInstance(PiuTimerAdapter.this.mActivity).playPositionTimer(i, PiuTimerAdapter.this.showPiuTimerMode);
            }
        });
        piuTimerViewHolder.img_delete.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.PiuTimerAdapter.2
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                PiuTimerAdapter.this.mList.remove(i);
                PiuTimerManager.getInstance(PiuTimerAdapter.this.mActivity).deleteShared(PiuTimerAdapter.this.mList, PiuTimerAdapter.this.showPiuTimerMode, i);
                PiuTimerAdapter.this.notifyItemRemoved(i);
                PiuTimerAdapter piuTimerAdapter = PiuTimerAdapter.this;
                piuTimerAdapter.notifyItemRangeChanged(i, piuTimerAdapter.mList.size());
            }
        });
        piuTimerViewHolder.edit_time.setOnFocusChangeListener(new View.OnFocusChangeListener() { // from class: com.hismith.hismithapp.adapter.PiuTimerAdapter.3
            @Override // android.view.View.OnFocusChangeListener
            public void onFocusChange(View view, boolean z) {
                LogUtil.e("textsssd onFocusChangeL:" + i + ", hasFocus:" + z);
                if (z) {
                    TextChangeSaveShared textChangeSaveShared = PiuTimerAdapter.this.new TextChangeSaveShared(piuTimerViewHolder.edit_time, i);
                    piuTimerViewHolder.edit_time.addTextChangedListener(textChangeSaveShared);
                    piuTimerViewHolder.edit_time.setTag(textChangeSaveShared);
                    return;
                }
                if (piuTimerViewHolder.edit_time.getTag() instanceof TextWatcher) {
                    piuTimerViewHolder.edit_time.removeTextChangedListener((TextWatcher) piuTimerViewHolder.edit_time.getTag());
                }
                if (PiuTimerManager.getInstance(PiuTimerAdapter.this.mActivity).getmPlayLocalMode() == PiuTimerAdapter.this.showPiuTimerMode && PiuTimerManager.getInstance(PiuTimerAdapter.this.mActivity).getPlayPosition() == i && PiuTimerManager.getInstance(PiuTimerAdapter.this.mActivity).isPlay()) {
                    return;
                }
                LogUtil.e("textsssd 触发修改：" + PiuTimerManager.getInstance(PiuTimerAdapter.this.mActivity).getPlayPosition() + "," + i);
                try {
                    if (TextUtils.isEmpty(piuTimerViewHolder.edit_time.getText())) {
                        piuTimerViewHolder.edit_time.setText("0");
                    }
                    String obj = piuTimerViewHolder.edit_time.getText().toString();
                    ((PiuTimerModel) PiuTimerAdapter.this.mList.get(i)).setSecond(!obj.isEmpty() ? Integer.parseInt(obj) : 0);
                    PiuTimerManager.getInstance(PiuTimerAdapter.this.mActivity).saveShared(PiuTimerAdapter.this.mList, PiuTimerAdapter.this.showPiuTimerMode);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }

    /* renamed from: onBindViewHolder, reason: avoid collision after fix types in other method */
    public void onBindViewHolder2(PiuTimerViewHolder piuTimerViewHolder, int i, List<Object> list) {
        LogUtil.e("倒计时动画 onBindViewHolder position:" + i + ", payloads.get(0):" + list.toString());
        if (list.isEmpty()) {
            onBindViewHolder(piuTimerViewHolder, i);
        } else if (list.get(0).toString().startsWith("down")) {
            String[] split = list.get(0).toString().split(",");
            if (split.length > 1) {
                piuTimerViewHolder.edit_time.setText(split[1]);
            }
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public PiuTimerViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new PiuTimerViewHolder(LayoutInflater.from(this.mActivity).inflate(R.layout.item_piu_timer, viewGroup, false));
    }

    public void setShowPiuTimerMode(int i) {
        this.showPiuTimerMode = i;
    }
}
