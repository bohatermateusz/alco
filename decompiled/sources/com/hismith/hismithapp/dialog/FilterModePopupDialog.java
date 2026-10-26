package com.hismith.hismithapp.dialog;

import android.content.Context;
import android.text.TextUtils;
import android.view.MotionEvent;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import androidx.constraintlayout.core.motion.utils.TypedValues;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.Guideline;
import androidx.media3.extractor.text.ttml.TtmlNode;
import com.facebook.devicerequests.internal.DeviceRequestsHelper;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.lxj.xpopup.core.DrawerPopupView;
import com.lxj.xpopup.util.XPopupUtils;
import java.util.regex.Pattern;

/* loaded from: classes3.dex */
public class FilterModePopupDialog extends DrawerPopupView implements View.OnClickListener {
    private View.OnTouchListener SpeedMaxOnTouch;
    private float downX;
    private EditText edit_end_time;
    private EditText edit_name;
    private EditText edit_play_count_end;
    private EditText edit_play_count_start;
    private EditText edit_start_time;
    private String filterModeName;
    private String filterModePlayCount;
    private int filterModeSpeed;
    private String filterModeTime;
    private int filterOnlyThisProduct;
    private String filterPublishDate;
    private ImageView img_max_speed;
    private ImageView img_sort_play_count;
    private ImageView img_sort_time;
    private ConstraintLayout layout_max_speed;
    private ConstraintLayout layout_sort_play_count;
    private ConstraintLayout layout_sort_time;
    private Guideline line_max_start;
    private Context mContext;
    private OnFilterChangeListener onFilterChangeListener;
    private ProgressBar progress_speed;
    private ScrollView scroll_view;
    private String sortModeTime;
    private String sortPlayCount;
    private TextView txt_1_week;
    private TextView txt_24_hours;
    private TextView txt_3_days;
    private TextView txt_confirm;
    private TextView txt_device_label;
    private TextView txt_filter_all;
    private TextView txt_filter_this;
    private TextView txt_filter_title;
    private TextView txt_max_speed;
    private TextView txt_name_label;
    private TextView txt_play_count_label;
    private TextView txt_release_time_label;
    private TextView txt_reset;
    private TextView txt_sort_play_count;
    private TextView txt_sort_time;
    private TextView txt_sort_title;
    private TextView txt_speed_label;
    private TextView txt_time_label;

    public interface OnFilterChangeListener {
        void onFilterChange(int i, String str, String str2, int i2, String str3, String str4, String str5, String str6);
    }

    public FilterModePopupDialog(Context context, OnFilterChangeListener onFilterChangeListener) {
        super(context);
        this.filterOnlyThisProduct = 0;
        this.filterModeName = null;
        this.filterModeTime = null;
        this.filterModeSpeed = 100;
        this.filterModePlayCount = "";
        this.filterPublishDate = "";
        this.sortModeTime = "";
        this.sortPlayCount = "";
        this.SpeedMaxOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.dialog.FilterModePopupDialog.1
            /* JADX WARN: Code restructure failed: missing block: B:8:0x000e, code lost:
            
                if (r6 != 3) goto L24;
             */
            @Override // android.view.View.OnTouchListener
            /*
                Code decompiled incorrectly, please refer to instructions dump.
            */
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action != 0) {
                    if (action != 1) {
                        if (action == 2) {
                            float x = motionEvent.getX() - FilterModePopupDialog.this.downX;
                            int width = FilterModePopupDialog.this.progress_speed.getWidth();
                            int right = (int) (FilterModePopupDialog.this.line_max_start.getRight() - (FilterModePopupDialog.this.layout_max_speed.getRight() + x));
                            LogUtil.e("SpeedMaxOnTouch ACTION_MOVE:" + right);
                            if (right < 0) {
                                right = 0;
                            } else if (right > width) {
                                right = width;
                            }
                            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) FilterModePopupDialog.this.layout_max_speed.getLayoutParams();
                            layoutParams.setMargins(0, 0, right, 0);
                            FilterModePopupDialog.this.layout_max_speed.setLayoutParams(layoutParams);
                            int i = (int) (100.0d - ((right * 100.0d) / width));
                            FilterModePopupDialog.this.filterModeSpeed = i;
                            FilterModePopupDialog.this.txt_max_speed.setText(i + "");
                            FilterModePopupDialog.this.progress_speed.setProgress(i);
                        }
                    }
                    if (FilterModePopupDialog.this.scroll_view != null) {
                        FilterModePopupDialog.this.scroll_view.requestDisallowInterceptTouchEvent(false);
                    }
                } else {
                    if (FilterModePopupDialog.this.scroll_view != null) {
                        FilterModePopupDialog.this.scroll_view.requestDisallowInterceptTouchEvent(true);
                    }
                    FilterModePopupDialog.this.downX = motionEvent.getX();
                    LogUtil.e("SpeedMaxOnTouch ACTION_DOWN:" + FilterModePopupDialog.this.downX);
                }
                return true;
            }
        };
        this.mContext = context;
        this.onFilterChangeListener = onFilterChangeListener;
    }

    private void clickConfirmFilter() {
        this.filterModeName = this.edit_name.getText().toString();
        try {
            boolean isNumeric = isNumeric(this.edit_start_time.getText().toString());
            EditText editText = this.edit_start_time;
            String obj = isNumeric ? editText.getText().toString() : editText.getText().toString();
            boolean isNumeric2 = isNumeric(this.edit_end_time.getText().toString());
            EditText editText2 = this.edit_end_time;
            this.filterModeTime = obj + "-" + (isNumeric2 ? editText2.getText().toString() : editText2.getText().toString());
        } catch (Exception e) {
            e.printStackTrace();
        }
        try {
            boolean isNumeric3 = isNumeric(this.edit_play_count_start.getText().toString());
            EditText editText3 = this.edit_play_count_start;
            String obj2 = isNumeric3 ? editText3.getText().toString() : editText3.getText().toString();
            boolean isNumeric4 = isNumeric(this.edit_play_count_end.getText().toString());
            EditText editText4 = this.edit_play_count_end;
            this.filterModePlayCount = obj2 + "-" + (isNumeric4 ? editText4.getText().toString() : editText4.getText().toString());
        } catch (Exception e2) {
            e2.printStackTrace();
        }
        OnFilterChangeListener onFilterChangeListener = this.onFilterChangeListener;
        if (onFilterChangeListener != null) {
            onFilterChangeListener.onFilterChange(this.filterOnlyThisProduct, this.filterModeName, this.filterModeTime, this.filterModeSpeed, this.filterModePlayCount, this.filterPublishDate, this.sortModeTime, this.sortPlayCount);
        }
    }

    private void clickResetFilter() {
        refreshFilterAllOrThis(0);
        this.filterModeName = null;
        this.edit_name.setText("");
        this.filterModeTime = null;
        this.edit_start_time.setText("");
        this.edit_end_time.setText("");
        refreshSpeedMaxUI(100);
        this.filterModePlayCount = null;
        this.edit_play_count_start.setText("");
        this.edit_play_count_end.setText("");
        refreshFilterReleaseTime(null);
        refreshSortTime(null);
        refreshSortPlayCount(null);
        OnFilterChangeListener onFilterChangeListener = this.onFilterChangeListener;
        if (onFilterChangeListener != null) {
            onFilterChangeListener.onFilterChange(this.filterOnlyThisProduct, this.filterModeName, this.filterModeTime, this.filterModeSpeed, this.filterModePlayCount, this.filterPublishDate, this.sortModeTime, this.sortPlayCount);
        }
    }

    private void initData() {
        LanguageUtil.setText(this.mContext, this.txt_filter_title, "modeFilter");
        this.txt_device_label.setText(LanguageUtil.getLanguageByKey(this.mContext, DeviceRequestsHelper.DEVICE_INFO_DEVICE) + ":");
        LanguageUtil.setText(this.mContext, this.txt_filter_all, TtmlNode.COMBINE_ALL);
        LanguageUtil.setText(this.mContext, this.txt_filter_this, "current");
        this.txt_name_label.setText(LanguageUtil.getLanguageByKey(this.mContext, "modeName") + ":");
        this.txt_time_label.setText(LanguageUtil.getLanguageByKey(this.mContext, TypedValues.TransitionType.S_DURATION) + ":");
        this.txt_speed_label.setText(LanguageUtil.getLanguageByKey(this.mContext, "maxSpeed") + ":");
        this.txt_play_count_label.setText(LanguageUtil.getLanguageByKey(this.mContext, "playCount") + ":");
        this.txt_release_time_label.setText(LanguageUtil.getLanguageByKey(this.mContext, "releaseTime") + ":");
        LanguageUtil.setText(this.mContext, this.txt_24_hours, "oneDay");
        LanguageUtil.setText(this.mContext, this.txt_3_days, "threeDays");
        LanguageUtil.setText(this.mContext, this.txt_1_week, "oneWeek");
        LanguageUtil.setText(this.mContext, this.txt_sort_title, "sortBy");
        LanguageUtil.setText(this.mContext, this.txt_sort_time, TypedValues.TransitionType.S_DURATION);
        LanguageUtil.setText(this.mContext, this.txt_sort_play_count, "playCount");
        LanguageUtil.setHint(this.mContext, this.edit_start_time, "min");
        LanguageUtil.setHint(this.mContext, this.edit_end_time, "max");
        LanguageUtil.setHint(this.mContext, this.edit_play_count_start, "min");
        LanguageUtil.setHint(this.mContext, this.edit_play_count_end, "max");
        LanguageUtil.setText(this.mContext, this.txt_reset, "reset");
        LanguageUtil.setText(this.mContext, this.txt_confirm, "confirm");
    }

    private void initView() {
        this.scroll_view = (ScrollView) findViewById(R.id.scroll_view);
        this.txt_filter_title = (TextView) findViewById(R.id.txt_filter_title);
        this.txt_device_label = (TextView) findViewById(R.id.txt_device_label);
        this.txt_name_label = (TextView) findViewById(R.id.txt_name_label);
        this.txt_time_label = (TextView) findViewById(R.id.txt_time_label);
        this.txt_speed_label = (TextView) findViewById(R.id.txt_speed_label);
        this.txt_play_count_label = (TextView) findViewById(R.id.txt_play_count_label);
        this.txt_release_time_label = (TextView) findViewById(R.id.txt_release_time_label);
        this.txt_sort_title = (TextView) findViewById(R.id.txt_sort_title);
        this.txt_sort_time = (TextView) findViewById(R.id.txt_sort_time);
        this.txt_sort_play_count = (TextView) findViewById(R.id.txt_sort_play_count);
        this.txt_filter_all = (TextView) findViewById(R.id.txt_filter_all);
        this.txt_filter_this = (TextView) findViewById(R.id.txt_filter_this);
        this.txt_24_hours = (TextView) findViewById(R.id.txt_24_hours);
        this.txt_3_days = (TextView) findViewById(R.id.txt_3_days);
        this.txt_1_week = (TextView) findViewById(R.id.txt_1_week);
        this.edit_name = (EditText) findViewById(R.id.edit_name);
        this.edit_start_time = (EditText) findViewById(R.id.edit_start_time);
        this.edit_end_time = (EditText) findViewById(R.id.edit_end_time);
        this.edit_play_count_start = (EditText) findViewById(R.id.edit_play_count_start);
        this.edit_play_count_end = (EditText) findViewById(R.id.edit_play_count_end);
        this.progress_speed = (ProgressBar) findViewById(R.id.progress_speed);
        this.txt_max_speed = (TextView) findViewById(R.id.txt_max_speed);
        this.layout_max_speed = (ConstraintLayout) findViewById(R.id.layout_max_speed);
        this.img_max_speed = (ImageView) findViewById(R.id.img_max_speed);
        this.line_max_start = (Guideline) findViewById(R.id.line_max_start);
        this.layout_sort_time = (ConstraintLayout) findViewById(R.id.layout_sort_time);
        this.layout_sort_play_count = (ConstraintLayout) findViewById(R.id.layout_sort_play_count);
        this.img_sort_time = (ImageView) findViewById(R.id.img_sort_time);
        this.img_sort_play_count = (ImageView) findViewById(R.id.img_sort_play_count);
        this.txt_reset = (TextView) findViewById(R.id.txt_reset);
        this.txt_confirm = (TextView) findViewById(R.id.txt_confirm);
        if (TextUtils.isEmpty(this.filterModeName)) {
            return;
        }
        this.edit_name.setText(this.filterModeName);
    }

    private boolean isNumeric(String str) {
        return Pattern.compile("[0-9]*").matcher(str).matches();
    }

    private void refreshFilterAllOrThis(int i) {
        this.filterOnlyThisProduct = i;
        TextView textView = this.txt_filter_all;
        if (i == 0) {
            textView.setTextColor(getResources().getColor(R.color.app_style_color));
            this.txt_filter_all.setBackgroundResource(R.drawable.bg_corners_filter_button_style_18);
            this.txt_filter_this.setTextColor(getResources().getColor(R.color.filter_text_color));
            this.txt_filter_this.setBackground(null);
            return;
        }
        textView.setTextColor(getResources().getColor(R.color.filter_text_color));
        this.txt_filter_all.setBackground(null);
        this.txt_filter_this.setTextColor(getResources().getColor(R.color.app_style_color));
        this.txt_filter_this.setBackgroundResource(R.drawable.bg_corners_filter_button_style_18);
    }

    private void refreshFilterReleaseTime(String str) {
        this.filterPublishDate = str;
        if ("24Hour".equalsIgnoreCase(str)) {
            this.txt_24_hours.setTextColor(getResources().getColor(R.color.app_style_color));
            this.txt_24_hours.setBackgroundResource(R.drawable.bg_corners_filter_button_style_18);
            this.txt_3_days.setTextColor(getResources().getColor(R.color.filter_text_color));
            this.txt_3_days.setBackgroundResource(R.drawable.bg_corners_1affffff_18);
            this.txt_1_week.setTextColor(getResources().getColor(R.color.filter_text_color));
            this.txt_1_week.setBackgroundResource(R.drawable.bg_corners_1affffff_18);
            return;
        }
        if ("3Day".equalsIgnoreCase(this.filterPublishDate)) {
            this.txt_24_hours.setTextColor(getResources().getColor(R.color.filter_text_color));
            this.txt_24_hours.setBackgroundResource(R.drawable.bg_corners_1affffff_18);
            this.txt_3_days.setTextColor(getResources().getColor(R.color.app_style_color));
            this.txt_3_days.setBackgroundResource(R.drawable.bg_corners_filter_button_style_18);
            this.txt_1_week.setTextColor(getResources().getColor(R.color.filter_text_color));
            this.txt_1_week.setBackgroundResource(R.drawable.bg_corners_1affffff_18);
            return;
        }
        if ("1Week".equalsIgnoreCase(this.filterPublishDate)) {
            this.txt_24_hours.setTextColor(getResources().getColor(R.color.filter_text_color));
            this.txt_24_hours.setBackgroundResource(R.drawable.bg_corners_1affffff_18);
            this.txt_3_days.setTextColor(getResources().getColor(R.color.filter_text_color));
            this.txt_3_days.setBackgroundResource(R.drawable.bg_corners_1affffff_18);
            this.txt_1_week.setTextColor(getResources().getColor(R.color.app_style_color));
            this.txt_1_week.setBackgroundResource(R.drawable.bg_corners_filter_button_style_18);
            return;
        }
        this.filterPublishDate = null;
        this.txt_24_hours.setTextColor(getResources().getColor(R.color.filter_text_color));
        this.txt_24_hours.setBackgroundResource(R.drawable.bg_corners_1affffff_18);
        this.txt_3_days.setTextColor(getResources().getColor(R.color.filter_text_color));
        this.txt_3_days.setBackgroundResource(R.drawable.bg_corners_1affffff_18);
        this.txt_1_week.setTextColor(getResources().getColor(R.color.filter_text_color));
        this.txt_1_week.setBackgroundResource(R.drawable.bg_corners_1affffff_18);
    }

    private void refreshSortPlayCount(String str) {
        this.sortPlayCount = str;
        if ("up".equalsIgnoreCase(str)) {
            this.layout_sort_play_count.setBackgroundResource(R.drawable.bg_corners_filter_button_style_18);
            this.txt_sort_play_count.setTextColor(getResources().getColor(R.color.app_style_color));
            this.img_sort_play_count.setImageResource(R.mipmap.ic_sort_up);
            return;
        }
        boolean equalsIgnoreCase = "down".equalsIgnoreCase(this.sortPlayCount);
        ConstraintLayout constraintLayout = this.layout_sort_play_count;
        if (equalsIgnoreCase) {
            constraintLayout.setBackgroundResource(R.drawable.bg_corners_filter_button_style_18);
            this.txt_sort_play_count.setTextColor(getResources().getColor(R.color.app_style_color));
            this.img_sort_play_count.setImageResource(R.mipmap.ic_sort_down);
        } else {
            constraintLayout.setBackgroundResource(R.drawable.bg_corners_1affffff_18);
            this.txt_sort_play_count.setTextColor(getResources().getColor(R.color.filter_text_color));
            this.img_sort_play_count.setImageResource(R.mipmap.ic_sort_normal);
        }
    }

    private void refreshSortTime(String str) {
        this.sortModeTime = str;
        if ("up".equalsIgnoreCase(str)) {
            this.layout_sort_time.setBackgroundResource(R.drawable.bg_corners_filter_button_style_18);
            this.txt_sort_time.setTextColor(getResources().getColor(R.color.app_style_color));
            this.img_sort_time.setImageResource(R.mipmap.ic_sort_up);
            return;
        }
        boolean equalsIgnoreCase = "down".equalsIgnoreCase(this.sortModeTime);
        ConstraintLayout constraintLayout = this.layout_sort_time;
        if (equalsIgnoreCase) {
            constraintLayout.setBackgroundResource(R.drawable.bg_corners_filter_button_style_18);
            this.txt_sort_time.setTextColor(getResources().getColor(R.color.app_style_color));
            this.img_sort_time.setImageResource(R.mipmap.ic_sort_down);
        } else {
            constraintLayout.setBackgroundResource(R.drawable.bg_corners_1affffff_18);
            this.txt_sort_time.setTextColor(getResources().getColor(R.color.filter_text_color));
            this.img_sort_time.setImageResource(R.mipmap.ic_sort_normal);
        }
    }

    private void refreshSpeedMaxUI(int i) {
        if (this.filterModeSpeed != i) {
            this.filterModeSpeed = i;
            this.txt_max_speed.setText(this.filterModeSpeed + "");
            this.progress_speed.setProgress(this.filterModeSpeed);
            double width = this.progress_speed.getWidth();
            int i2 = (int) (width - ((width / 100.0d) * this.filterModeSpeed));
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.layout_max_speed.getLayoutParams();
            layoutParams.setMargins(0, 0, i2, 0);
            this.layout_max_speed.setLayoutParams(layoutParams);
        }
    }

    private void setListener() {
        this.txt_filter_all.setOnClickListener(this);
        this.txt_filter_this.setOnClickListener(this);
        this.txt_24_hours.setOnClickListener(this);
        this.txt_3_days.setOnClickListener(this);
        this.txt_1_week.setOnClickListener(this);
        this.layout_sort_time.setOnClickListener(this);
        this.layout_sort_play_count.setOnClickListener(this);
        this.layout_max_speed.setOnTouchListener(this.SpeedMaxOnTouch);
        this.txt_reset.setOnClickListener(this);
        this.txt_confirm.setOnClickListener(this);
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected int getImplLayoutId() {
        return R.layout.dialog_filter_mode;
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected int getMaxWidth() {
        return (int) (XPopupUtils.getScreenWidth(getContext()) * 0.88f);
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected int getPopupWidth() {
        return (int) (XPopupUtils.getScreenWidth(getContext()) * 0.88f);
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.txt_filter_all) {
            refreshFilterAllOrThis(0);
            return;
        }
        if (id == R.id.txt_filter_this) {
            refreshFilterAllOrThis(1);
            return;
        }
        if (id == R.id.txt_24_hours) {
            if ("24Hour".equalsIgnoreCase(this.filterPublishDate)) {
                refreshFilterReleaseTime(null);
                return;
            } else {
                refreshFilterReleaseTime("24Hour");
                return;
            }
        }
        if (id == R.id.txt_3_days) {
            if ("3Day".equalsIgnoreCase(this.filterPublishDate)) {
                refreshFilterReleaseTime(null);
                return;
            } else {
                refreshFilterReleaseTime("3Day");
                return;
            }
        }
        if (id == R.id.txt_1_week) {
            if ("1Week".equalsIgnoreCase(this.filterPublishDate)) {
                refreshFilterReleaseTime(null);
                return;
            } else {
                refreshFilterReleaseTime("1Week");
                return;
            }
        }
        if (id == R.id.layout_sort_time) {
            if (TextUtils.isEmpty(this.sortModeTime)) {
                refreshSortTime("up");
                return;
            } else if ("up".equalsIgnoreCase(this.sortModeTime)) {
                refreshSortTime("down");
                return;
            } else {
                if ("down".equalsIgnoreCase(this.sortModeTime)) {
                    refreshSortTime(null);
                    return;
                }
                return;
            }
        }
        if (id != R.id.layout_sort_play_count) {
            if (id == R.id.txt_reset) {
                clickResetFilter();
                dismiss();
                return;
            } else {
                if (id == R.id.txt_confirm) {
                    clickConfirmFilter();
                    dismiss();
                    return;
                }
                return;
            }
        }
        if (TextUtils.isEmpty(this.sortPlayCount)) {
            refreshSortPlayCount("up");
        } else if ("up".equalsIgnoreCase(this.sortPlayCount)) {
            refreshSortPlayCount("down");
        } else if ("down".equalsIgnoreCase(this.sortPlayCount)) {
            refreshSortPlayCount(null);
        }
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected void onCreate() {
        super.onCreate();
        initView();
        initData();
        setListener();
    }

    public void setFilterModeName(String str) {
        this.filterModeName = str;
        EditText editText = this.edit_name;
        if (editText != null) {
            editText.setText(str);
        }
    }
}
