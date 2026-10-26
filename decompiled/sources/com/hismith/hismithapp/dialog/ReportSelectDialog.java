package com.hismith.hismithapp.dialog;

import android.content.Context;
import android.text.Editable;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.media3.extractor.ts.PsExtractor;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.ReportSelectAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.lxj.xpopup.XPopup;
import com.lxj.xpopup.core.BottomPopupView;
import com.lxj.xpopup.util.XPopupUtils;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public abstract class ReportSelectDialog extends BottomPopupView implements View.OnClickListener {
    private ReportSelectAdapter mAdapter;
    private Button mBtnSubmit;
    private Context mContext;
    private Gson mGson;
    private List<String> mKeys;
    private LinearLayoutManager mLayoutManager;
    private List<String> mList;
    private RecyclerView mRecyclerView;
    private TextView mTxtTitle;
    private int maxLength;
    private EditText report_edit_content;
    private LinearLayout report_layout_content;
    private TextView report_txt_count;
    private TextView report_txt_label;

    public ReportSelectDialog(Context context) {
        super(context);
        this.maxLength = PsExtractor.VIDEO_STREAM_MASK;
        this.mContext = context;
        this.mGson = new Gson();
    }

    public abstract void btnSubmit(String str, String str2);

    @Override // com.lxj.xpopup.core.BottomPopupView, com.lxj.xpopup.core.BasePopupView
    protected int getImplLayoutId() {
        return R.layout.dialog_report_select;
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected int getMaxHeight() {
        return (int) (XPopupUtils.getScreenHeight(getContext()) * 0.9f);
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() != R.id.report_select_btn_submit || this.mAdapter.getClickPosition() == -1) {
            return;
        }
        btnSubmit(this.mKeys.get(this.mAdapter.getClickPosition()), this.report_edit_content.getText().toString().trim());
        dismiss();
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected void onCreate() {
        super.onCreate();
        this.mTxtTitle = (TextView) findViewById(R.id.report_select_txt_title);
        this.mRecyclerView = (RecyclerView) findViewById(R.id.report_select_recyclerView);
        this.report_txt_label = (TextView) findViewById(R.id.report_txt_label);
        this.report_layout_content = (LinearLayout) findViewById(R.id.report_layout_content);
        this.report_edit_content = (EditText) findViewById(R.id.report_edit_content);
        this.report_txt_count = (TextView) findViewById(R.id.report_txt_count);
        this.mBtnSubmit = (Button) findViewById(R.id.report_select_btn_submit);
        this.report_txt_label.setText(LanguageUtil.getLanguageByKey(this.mContext, "reportReason") + ":");
        LanguageUtil.setText(this.mContext, this.mTxtTitle, "reportReason");
        LanguageUtil.setText(this.mContext, this.mBtnSubmit, "submit");
        this.report_txt_count.setText(this.report_edit_content.getText().length() + "/" + this.maxLength);
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.mContext);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mKeys = new ArrayList();
        this.mList = new ArrayList();
        String str = (String) SharedPreferencesUtil.get(this.mContext, Constans.complaintReasons);
        if (TextUtils.isEmpty(str)) {
            MyApplication.getApplication().complaintReasons((String) SharedPreferencesUtil.get(this.mContext, Constans.Token));
        } else {
            JsonObject jsonObject = (JsonObject) this.mGson.fromJson(str, JsonObject.class);
            for (String str2 : jsonObject.getAsJsonObject().keySet()) {
                String asString = jsonObject.getAsJsonObject().get(str2).getAsString();
                this.mKeys.add(str2);
                this.mList.add(asString);
            }
        }
        ReportSelectAdapter reportSelectAdapter = new ReportSelectAdapter(this.mContext, this.mList);
        this.mAdapter = reportSelectAdapter;
        this.mRecyclerView.setAdapter(reportSelectAdapter);
        this.mBtnSubmit.setOnClickListener(this);
        this.report_edit_content.addTextChangedListener(new TextWatcher() { // from class: com.hismith.hismithapp.dialog.ReportSelectDialog.1
            @Override // android.text.TextWatcher
            public void afterTextChanged(Editable editable) {
                ReportSelectDialog.this.report_txt_count.setText(ReportSelectDialog.this.report_edit_content.getText().length() + "/" + ReportSelectDialog.this.maxLength);
            }

            @Override // android.text.TextWatcher
            public void beforeTextChanged(CharSequence charSequence, int i, int i2, int i3) {
            }

            @Override // android.text.TextWatcher
            public void onTextChanged(CharSequence charSequence, int i, int i2, int i3) {
            }
        });
    }

    public void showDialog() {
        new XPopup.Builder(this.mContext).autoFocusEditText(false).autoOpenSoftInput(false).asCustom(this).show();
    }
}
