package com.hismith.hismithapp.dialog;

import android.app.Activity;
import android.app.Dialog;
import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.LineUserAdapter;
import com.hismith.hismithapp.layoutmanager.WrapContentLinearLayoutManager;
import com.hismith.hismithapp.model.socket.SocketUserModel;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class LineUserDialog extends Dialog implements View.OnClickListener {
    private Button btn_cancel;
    private Button btn_complete;
    private Activity mActivity;
    private LineUserAdapter mAdapter;
    private int mHallUserFlag;
    private WrapContentLinearLayoutManager mLayoutManager;
    private List<SocketUserModel> mList;
    private String mTitleId;
    private RecyclerView recyclerView;
    private SmartRefreshLayout refreshLayout;
    private ImageView title_img_back;
    private TextView txt_title;

    public LineUserDialog(Activity activity, List<SocketUserModel> list, String str, int i) {
        super(activity);
        this.mActivity = activity;
        this.mList = list;
        this.mTitleId = str;
        this.mHallUserFlag = i;
    }

    private void setListener() {
        this.title_img_back.setOnClickListener(this);
    }

    public void addUser(List<SocketUserModel> list) {
        if (this.mList == null || this.mAdapter == null) {
            return;
        }
        LogUtil.e("SocketManager notifyItemInserted (mList.size()-socketUserModel.size()):（" + this.mList.size() + "-" + list.size() + ")=" + (this.mList.size() - list.size()));
        this.mAdapter.notifyItemInserted(this.mList.size() - list.size());
    }

    @Override // android.app.Dialog, android.view.Window.Callback
    public void onAttachedToWindow() {
        super.onAttachedToWindow();
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (R.id.title_img_back == view.getId()) {
            dismiss();
        }
    }

    @Override // android.app.Dialog
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.dialog_line_user);
        Window window = getWindow();
        if (window != null) {
            window.setBackgroundDrawable(null);
            WindowManager.LayoutParams attributes = window.getAttributes();
            int width = this.mActivity.getWindowManager().getDefaultDisplay().getWidth();
            int height = this.mActivity.getWindowManager().getDefaultDisplay().getHeight();
            LogUtil.e("sreenHeight:" + height);
            attributes.width = (int) (width * 0.9d);
            attributes.height = (int) (height * 0.7d);
            window.setAttributes(attributes);
        }
        this.title_img_back = (ImageView) findViewById(R.id.title_img_back);
        this.txt_title = (TextView) findViewById(R.id.txt_title);
        this.btn_cancel = (Button) findViewById(R.id.btn_cancel);
        this.btn_complete = (Button) findViewById(R.id.btn_complete);
        LanguageUtil.setText(this.mActivity, this.txt_title, this.mTitleId);
        LanguageUtil.setText(this.mActivity, this.btn_complete, "confirm");
        LanguageUtil.setText(this.mActivity, this.btn_cancel, "cancel");
        this.refreshLayout = (SmartRefreshLayout) findViewById(R.id.refreshLayout);
        RecyclerView recyclerView = (RecyclerView) findViewById(R.id.recycler_view);
        this.recyclerView = recyclerView;
        WrapContentLinearLayoutManager wrapContentLinearLayoutManager = new WrapContentLinearLayoutManager(this.mActivity);
        this.mLayoutManager = wrapContentLinearLayoutManager;
        recyclerView.setLayoutManager(wrapContentLinearLayoutManager);
        if (this.mList == null) {
            this.mList = new ArrayList();
        }
        LineUserAdapter lineUserAdapter = new LineUserAdapter(this.mActivity, this.mList, this.mHallUserFlag);
        this.mAdapter = lineUserAdapter;
        this.recyclerView.setAdapter(lineUserAdapter);
        this.refreshLayout.setEnableLoadMore(false);
        this.refreshLayout.setEnableRefresh(false);
        setCanceledOnTouchOutside(true);
        setListener();
    }

    @Override // android.app.Dialog, android.view.Window.Callback
    public void onDetachedFromWindow() {
        super.onDetachedFromWindow();
    }

    public void removeUser(int i) {
        if (this.mAdapter != null) {
            LogUtil.e("SocketManager notifyItemRemoved:" + i);
            this.mAdapter.notifyItemRemoved(i);
        }
    }

    @Override // android.app.Dialog
    public void show() {
        super.show();
    }
}
