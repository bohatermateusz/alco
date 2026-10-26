package com.hismith.hismithapp.dialog;

import android.app.Activity;
import android.text.Editable;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.Group;
import androidx.recyclerview.widget.ItemTouchHelper;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.google.gson.Gson;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.PiuTimerAdapter;
import com.hismith.hismithapp.adapter.callback.PiuItemDragCallback;
import com.hismith.hismithapp.listener.SoftKeyBoardListener;
import com.hismith.hismithapp.manager.PiuTimerManager;
import com.hismith.hismithapp.model.PiuTimerModel;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.lxj.xpopup.XPopup;
import com.lxj.xpopup.core.BottomPopupView;
import com.lxj.xpopup.util.XPopupUtils;
import java.util.List;

/* loaded from: classes3.dex */
public class PiuTimerDialog extends BottomPopupView implements View.OnClickListener {
    private EditText edit_max_time;
    private EditText edit_min_time;
    private Group group_list;
    private Gson gson;
    private ImageView img_ic_pause;
    private ImageView img_random_play_bg;
    private boolean isPlay;
    private ConstraintLayout layout_item;
    private LinearLayout layout_pause;
    private ConstraintLayout layout_random;
    private Activity mActivity;
    private PiuTimerAdapter mAdapter;
    private float mHeightPercent;
    private ItemTouchHelper mHelper;
    private LinearLayoutManager mLayoutManager;
    private List<PiuTimerModel> mList;
    private RecyclerView recycler_timer;
    private int showPiuTimerMode;
    private SoftKeyBoardListener softKeyBoardListener;
    private ImageView title_img_back;
    private TextView txt_add;
    private TextView txt_one;
    private TextView txt_pause;
    private TextView txt_random;
    private TextView txt_random_title;
    private TextView txt_three;
    private TextView txt_title;
    private TextView txt_two;

    class TextChangeRandomTime implements TextWatcher {
        private EditText editText;

        public TextChangeRandomTime(EditText editText) {
            this.editText = editText;
        }

        @Override // android.text.TextWatcher
        public void afterTextChanged(Editable editable) {
        }

        @Override // android.text.TextWatcher
        public void beforeTextChanged(CharSequence charSequence, int i, int i2, int i3) {
        }

        @Override // android.text.TextWatcher
        public void onTextChanged(CharSequence charSequence, int i, int i2, int i3) {
            LogUtil.e("textsssd onTextChanged " + ((Object) charSequence) + ", start:" + i + ", before:" + i2 + ", count:" + i3);
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

    public PiuTimerDialog(Activity activity, float f) {
        super(activity);
        this.isPlay = false;
        this.mActivity = activity;
        this.mHeightPercent = f;
    }

    private void changePlayStatus(boolean z) {
        PiuTimerAdapter piuTimerAdapter;
        PiuTimerAdapter piuTimerAdapter2;
        Activity activity = this.mActivity;
        if (z) {
            LanguageUtil.setText(activity, this.txt_pause, "pause");
            this.img_ic_pause.setImageResource(R.mipmap.ic_play_pause);
            int i = PiuTimerManager.getInstance(this.mActivity).getmPlayLocalMode();
            ImageView imageView = this.img_random_play_bg;
            if (i == 4) {
                imageView.setVisibility(0);
            } else {
                imageView.setVisibility(4);
            }
            if (PiuTimerManager.getInstance(this.mActivity).getmPlayLocalMode() == 4 || (piuTimerAdapter2 = this.mAdapter) == null) {
                return;
            }
            piuTimerAdapter2.notifyDataSetChanged();
            return;
        }
        LanguageUtil.setText(activity, this.txt_pause, "play");
        this.img_ic_pause.setImageResource(R.mipmap.ic_play_mode_preview);
        int i2 = PiuTimerManager.getInstance(this.mActivity).getmPlayLocalMode();
        ImageView imageView2 = this.img_random_play_bg;
        if (i2 == 4) {
            imageView2.setVisibility(0);
        } else {
            imageView2.setVisibility(4);
        }
        if (PiuTimerManager.getInstance(this.mActivity).getmPlayLocalMode() == 4 || (piuTimerAdapter = this.mAdapter) == null) {
            return;
        }
        piuTimerAdapter.notifyDataSetChanged();
    }

    private void initData() {
        this.gson = new Gson();
        this.softKeyBoardListener = new SoftKeyBoardListener(getActivity());
        this.showPiuTimerMode = 1;
        this.mList = PiuTimerManager.getInstance(this.mActivity).getmList(this.showPiuTimerMode);
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.mActivity);
        this.mLayoutManager = linearLayoutManager;
        this.recycler_timer.setLayoutManager(linearLayoutManager);
        PiuTimerAdapter piuTimerAdapter = new PiuTimerAdapter(this.mActivity, this.showPiuTimerMode, this.mList);
        this.mAdapter = piuTimerAdapter;
        this.recycler_timer.setAdapter(piuTimerAdapter);
        ItemTouchHelper itemTouchHelper = new ItemTouchHelper(new PiuItemDragCallback(this.mActivity, this.mAdapter, this.showPiuTimerMode, this.mList));
        this.mHelper = itemTouchHelper;
        itemTouchHelper.attachToRecyclerView(this.recycler_timer);
        boolean isPlay = PiuTimerManager.getInstance(this.mActivity).isPlay();
        this.isPlay = isPlay;
        changePlayStatus(isPlay);
    }

    private void initView() {
        this.title_img_back = (ImageView) findViewById(R.id.title_img_back);
        this.txt_title = (TextView) findViewById(R.id.txt_title);
        this.txt_one = (TextView) findViewById(R.id.txt_one);
        this.txt_two = (TextView) findViewById(R.id.txt_two);
        this.txt_three = (TextView) findViewById(R.id.txt_three);
        this.txt_random = (TextView) findViewById(R.id.txt_random);
        this.recycler_timer = (RecyclerView) findViewById(R.id.recycler_timer);
        this.layout_pause = (LinearLayout) findViewById(R.id.layout_pause);
        this.img_ic_pause = (ImageView) findViewById(R.id.img_ic_pause);
        this.txt_pause = (TextView) findViewById(R.id.txt_pause);
        this.txt_add = (TextView) findViewById(R.id.txt_add);
        this.group_list = (Group) findViewById(R.id.group_list);
        this.layout_random = (ConstraintLayout) findViewById(R.id.layout_random);
        this.img_random_play_bg = (ImageView) findViewById(R.id.img_random_play_bg);
        this.layout_item = (ConstraintLayout) findViewById(R.id.layout_item);
        this.txt_random_title = (TextView) findViewById(R.id.txt_random_title);
        this.edit_min_time = (EditText) findViewById(R.id.edit_min_time);
        this.edit_max_time = (EditText) findViewById(R.id.edit_max_time);
        LanguageUtil.setText(this.mActivity, this.txt_title, "countdownSetting");
        LanguageUtil.setText(this.mActivity, this.txt_add, "add");
        LanguageUtil.setText(this.mActivity, this.txt_random, "random");
        this.txt_random_title.setText(LanguageUtil.getLanguageByKey(this.mActivity, "random_time") + ":");
        this.mActivity.getWindow().setSoftInputMode(32);
    }

    private void notifyList(int i) {
        this.showPiuTimerMode = i;
        if (i == 1) {
            this.txt_one.setTextColor(this.mActivity.getColor(R.color.piu_tab_select_text_color));
            this.txt_one.setBackgroundResource(R.drawable.bg_piu_tab_selected);
            this.txt_two.setTextColor(this.mActivity.getColor(R.color.piu_tab_normal_text_color));
            this.txt_two.setBackground(null);
            this.txt_three.setTextColor(this.mActivity.getColor(R.color.piu_tab_normal_text_color));
            this.txt_three.setBackground(null);
            this.txt_random.setTextColor(this.mActivity.getColor(R.color.piu_tab_normal_text_color));
            this.txt_random.setBackground(null);
            if (this.group_list.getVisibility() != 0) {
                this.group_list.setVisibility(0);
            }
            if (this.layout_random.getVisibility() != 8) {
                this.layout_random.setVisibility(8);
            }
        } else if (i == 2) {
            this.txt_one.setTextColor(this.mActivity.getColor(R.color.piu_tab_normal_text_color));
            this.txt_one.setBackground(null);
            this.txt_two.setTextColor(this.mActivity.getColor(R.color.piu_tab_select_text_color));
            this.txt_two.setBackgroundResource(R.drawable.bg_piu_tab_selected);
            this.txt_three.setTextColor(this.mActivity.getColor(R.color.piu_tab_normal_text_color));
            this.txt_three.setBackground(null);
            this.txt_random.setTextColor(this.mActivity.getColor(R.color.piu_tab_normal_text_color));
            this.txt_random.setBackground(null);
            if (this.group_list.getVisibility() != 0) {
                this.group_list.setVisibility(0);
            }
            if (this.layout_random.getVisibility() != 8) {
                this.layout_random.setVisibility(8);
            }
        } else if (i == 3) {
            this.txt_one.setTextColor(this.mActivity.getColor(R.color.piu_tab_normal_text_color));
            this.txt_one.setBackground(null);
            this.txt_two.setTextColor(this.mActivity.getColor(R.color.piu_tab_normal_text_color));
            this.txt_two.setBackground(null);
            this.txt_three.setTextColor(this.mActivity.getColor(R.color.piu_tab_select_text_color));
            this.txt_three.setBackgroundResource(R.drawable.bg_piu_tab_selected);
            this.txt_random.setTextColor(this.mActivity.getColor(R.color.piu_tab_normal_text_color));
            this.txt_random.setBackground(null);
            if (this.group_list.getVisibility() != 0) {
                this.group_list.setVisibility(0);
            }
            if (this.layout_random.getVisibility() != 8) {
                this.layout_random.setVisibility(8);
            }
        } else if (i == 4) {
            this.txt_one.setTextColor(this.mActivity.getColor(R.color.piu_tab_normal_text_color));
            this.txt_one.setBackground(null);
            this.txt_two.setTextColor(this.mActivity.getColor(R.color.piu_tab_normal_text_color));
            this.txt_two.setBackground(null);
            this.txt_three.setTextColor(this.mActivity.getColor(R.color.piu_tab_normal_text_color));
            this.txt_three.setBackground(null);
            this.txt_random.setTextColor(this.mActivity.getColor(R.color.piu_tab_select_text_color));
            this.txt_random.setBackgroundResource(R.drawable.bg_piu_tab_selected);
            if (this.group_list.getVisibility() != 8) {
                this.group_list.setVisibility(8);
            }
            if (this.layout_random.getVisibility() != 0) {
                this.layout_random.setVisibility(0);
            }
        }
        this.mList.clear();
        this.mList.addAll(PiuTimerManager.getInstance(this.mActivity).getmList(this.showPiuTimerMode));
        int i2 = this.showPiuTimerMode;
        if (i2 != 4) {
            PiuTimerAdapter piuTimerAdapter = this.mAdapter;
            if (piuTimerAdapter != null) {
                piuTimerAdapter.setShowPiuTimerMode(i2);
                this.mAdapter.notifyDataSetChanged();
                return;
            }
            return;
        }
        this.edit_min_time.setText(this.mList.get(0).getSecond() + "");
        this.edit_max_time.setText(this.mList.get(1).getSecond() + "");
        int i3 = PiuTimerManager.getInstance(this.mActivity).getmPlayLocalMode();
        ImageView imageView = this.img_random_play_bg;
        if (i3 == 4) {
            imageView.setVisibility(0);
        } else {
            imageView.setVisibility(4);
        }
    }

    private void setListener() {
        this.title_img_back.setOnClickListener(this);
        this.layout_pause.setOnClickListener(this);
        this.txt_add.setOnClickListener(this);
        this.txt_one.setOnClickListener(this);
        this.txt_two.setOnClickListener(this);
        this.txt_three.setOnClickListener(this);
        this.txt_random.setOnClickListener(this);
        this.layout_item.setOnClickListener(this);
        this.edit_min_time.addTextChangedListener(new TextChangeRandomTime(this.edit_min_time));
        this.edit_max_time.addTextChangedListener(new TextChangeRandomTime(this.edit_max_time));
        this.softKeyBoardListener.setListener(new SoftKeyBoardListener.OnSoftKeyBoardChangeListener() { // from class: com.hismith.hismithapp.dialog.PiuTimerDialog.1
            @Override // com.hismith.hismithapp.listener.SoftKeyBoardListener.OnSoftKeyBoardChangeListener
            public void keyBoardHide(int i) {
                int i2;
                int i3;
                if (PiuTimerDialog.this.showPiuTimerMode == 4) {
                    boolean isEmpty = TextUtils.isEmpty(PiuTimerDialog.this.edit_min_time.getText());
                    PiuTimerDialog piuTimerDialog = PiuTimerDialog.this;
                    if (isEmpty) {
                        piuTimerDialog.edit_min_time.setText("0");
                        i2 = 0;
                    } else {
                        i2 = Integer.parseInt(piuTimerDialog.edit_min_time.getText().toString());
                    }
                    boolean isEmpty2 = TextUtils.isEmpty(PiuTimerDialog.this.edit_max_time.getText());
                    PiuTimerDialog piuTimerDialog2 = PiuTimerDialog.this;
                    if (isEmpty2) {
                        piuTimerDialog2.edit_max_time.setText("0");
                        i3 = 0;
                    } else {
                        i3 = Integer.parseInt(piuTimerDialog2.edit_max_time.getText().toString());
                    }
                    ((PiuTimerModel) PiuTimerDialog.this.mList.get(0)).setSecond(i2);
                    ((PiuTimerModel) PiuTimerDialog.this.mList.get(1)).setSecond(i3);
                    PiuTimerManager.getInstance(PiuTimerDialog.this.mActivity).saveShared(PiuTimerDialog.this.mList, 4);
                }
            }

            @Override // com.hismith.hismithapp.listener.SoftKeyBoardListener.OnSoftKeyBoardChangeListener
            public void keyBoardShow(int i) {
            }
        });
        setSoftKeyBoardListener();
    }

    private void setSoftKeyBoardListener() {
        SoftKeyBoardListener softKeyBoardListener = new SoftKeyBoardListener(this.mActivity);
        this.softKeyBoardListener = softKeyBoardListener;
        softKeyBoardListener.setListener(new SoftKeyBoardListener.OnSoftKeyBoardChangeListener() { // from class: com.hismith.hismithapp.dialog.PiuTimerDialog.2
            @Override // com.hismith.hismithapp.listener.SoftKeyBoardListener.OnSoftKeyBoardChangeListener
            public void keyBoardHide(int i) {
                LogUtil.w("隐藏软键盘");
                int i2 = PiuTimerDialog.this.showPiuTimerMode;
                PiuTimerDialog piuTimerDialog = PiuTimerDialog.this;
                if (i2 != 4) {
                    piuTimerDialog.layout_pause.setVisibility(0);
                    PiuTimerDialog.this.txt_add.setVisibility(0);
                } else {
                    piuTimerDialog.layout_pause.setVisibility(0);
                    if (PiuTimerDialog.this.txt_add.getVisibility() != 8) {
                        PiuTimerDialog.this.txt_add.setVisibility(8);
                    }
                }
            }

            @Override // com.hismith.hismithapp.listener.SoftKeyBoardListener.OnSoftKeyBoardChangeListener
            public void keyBoardShow(int i) {
                LogUtil.w("显示软键盘");
                int i2 = PiuTimerDialog.this.showPiuTimerMode;
                PiuTimerDialog piuTimerDialog = PiuTimerDialog.this;
                if (i2 != 4) {
                    piuTimerDialog.layout_pause.setVisibility(8);
                    PiuTimerDialog.this.txt_add.setVisibility(8);
                } else {
                    piuTimerDialog.layout_pause.setVisibility(8);
                    if (PiuTimerDialog.this.txt_add.getVisibility() != 8) {
                        PiuTimerDialog.this.txt_add.setVisibility(8);
                    }
                }
            }
        });
    }

    public void downNotifyAdapter(int i) {
        LogUtil.e("倒计时动画 downNotifyAdapter mAdapter:" + this.mAdapter);
        if (this.mAdapter == null || PiuTimerManager.getInstance(this.mActivity).getmPlayLocalMode() != this.showPiuTimerMode) {
            return;
        }
        this.mAdapter.notifyItemChanged(PiuTimerManager.getInstance(this.mActivity).getPlayPosition(), "down," + i);
    }

    @Override // com.lxj.xpopup.core.BottomPopupView, com.lxj.xpopup.core.BasePopupView
    protected int getImplLayoutId() {
        return R.layout.dialog_piu_timer;
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected int getPopupHeight() {
        return (int) (XPopupUtils.getScreenHeight(getContext()) * this.mHeightPercent);
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            dismiss();
            return;
        }
        boolean z = false;
        if (id == R.id.layout_pause) {
            if (this.isPlay) {
                PiuTimerManager.getInstance(this.mActivity).pauseTimer();
            } else if (this.showPiuTimerMode != PiuTimerManager.getInstance(this.mActivity).getmPlayLocalMode()) {
                List<PiuTimerModel> list = this.mList;
                if (list != null && !list.isEmpty()) {
                    z = PiuTimerManager.getInstance(this.mActivity).playPositionTimer(0, this.showPiuTimerMode);
                }
            } else {
                z = PiuTimerManager.getInstance(this.mActivity).playTimer();
            }
            this.isPlay = z;
            return;
        }
        if (id == R.id.txt_add) {
            PiuTimerAdapter piuTimerAdapter = this.mAdapter;
            if (piuTimerAdapter != null) {
                piuTimerAdapter.addItem();
                this.recycler_timer.smoothScrollToPosition(this.mAdapter.getItemCount() - 1);
                return;
            }
            return;
        }
        if (id == R.id.txt_one) {
            if (this.showPiuTimerMode != 1) {
                notifyList(1);
                return;
            }
            return;
        }
        if (id == R.id.txt_two) {
            if (this.showPiuTimerMode != 2) {
                notifyList(2);
            }
        } else if (id == R.id.txt_three) {
            if (this.showPiuTimerMode != 3) {
                notifyList(3);
            }
        } else if (id == R.id.txt_random) {
            if (this.showPiuTimerMode != 4) {
                notifyList(4);
            }
        } else {
            if (id != R.id.layout_item || PiuTimerManager.getInstance(this.mActivity).getmPlayLocalMode() == 4) {
                return;
            }
            PiuTimerManager.getInstance(this.mActivity).playPositionTimer(0, 4);
        }
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected void onCreate() {
        super.onCreate();
        initView();
        initData();
        setListener();
    }

    public void setPlay(boolean z) {
        this.isPlay = z;
        changePlayStatus(z);
    }

    public void showDialog() {
        new XPopup.Builder(this.mActivity).autoFocusEditText(false).autoOpenSoftInput(false).moveUpToKeyboard(false).isLightNavigationBar(true).asCustom(this).show();
    }
}
