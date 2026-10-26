package com.hismith.hismithapp.dialog;

import android.content.Context;
import android.widget.TextView;
import androidx.recyclerview.widget.LinearLayoutManager;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.LineUserAdapter;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.model.socket.SocketUserModel;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.view.SlideRecyclerView;
import com.lxj.xpopup.XPopup;
import com.lxj.xpopup.core.BasePopupView;
import com.lxj.xpopup.core.BottomPopupView;
import com.lxj.xpopup.util.XPopupUtils;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class UserDrawerPopupDialog extends BottomPopupView {
    private RetrofitRequest baseRequest;
    private BasePopupView loadPopupView;
    private LineUserAdapter mAdapter;
    private Context mContext;
    private int mHallUserFlag;
    private LinearLayoutManager mLayoutManager;
    private List<SocketUserModel> mList;
    private SlideRecyclerView mRecyclerUser;
    private String mTitleId;
    private TextView mTitleName;
    private LineUserAdapter.OnItemClickListener onItemClickListener;

    public UserDrawerPopupDialog(Context context) {
        super(context);
        this.mHallUserFlag = 0;
        this.mContext = context;
    }

    public UserDrawerPopupDialog(Context context, String str, int i, LineUserAdapter.OnItemClickListener onItemClickListener) {
        super(context);
        this.mContext = context;
        this.mTitleId = str;
        this.mHallUserFlag = i;
        this.onItemClickListener = onItemClickListener;
    }

    public UserDrawerPopupDialog(Context context, List<SocketUserModel> list, String str, int i, LineUserAdapter.OnItemClickListener onItemClickListener) {
        super(context);
        this.mContext = context;
        this.mList = list;
        this.mTitleId = str;
        this.mHallUserFlag = i;
        this.onItemClickListener = onItemClickListener;
    }

    private void initData() {
        this.baseRequest = new RetrofitRequest();
        SlideRecyclerView slideRecyclerView = this.mRecyclerUser;
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.mContext);
        this.mLayoutManager = linearLayoutManager;
        slideRecyclerView.setLayoutManager(linearLayoutManager);
        this.mList = new ArrayList();
        LineUserAdapter lineUserAdapter = new LineUserAdapter(this.mContext, this.mList, this.mHallUserFlag);
        this.mAdapter = lineUserAdapter;
        lineUserAdapter.setOnItemClickListener(this.onItemClickListener);
        this.mRecyclerUser.setAdapter(this.mAdapter);
        this.loadPopupView = new XPopup.Builder(getContext()).asLoading(LanguageUtil.getLanguageByKey(getContext(), "isLoading")).show();
    }

    private void initView() {
        this.mTitleName = (TextView) findViewById(R.id.title_txt_name);
        this.mRecyclerUser = (SlideRecyclerView) findViewById(R.id.dialog_drawer_recycler_user);
        LanguageUtil.setText(this.mContext, this.mTitleName, this.mTitleId);
    }

    private void setListener() {
    }

    @Override // com.lxj.xpopup.core.BottomPopupView, com.lxj.xpopup.core.BasePopupView
    protected int getImplLayoutId() {
        return R.layout.dialog_drawer_user;
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected int getMaxHeight() {
        return (int) (XPopupUtils.getScreenHeight(getContext()) * 0.8f);
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected void onCreate() {
        super.onCreate();
        initView();
        initData();
        setListener();
    }

    public void setUserList(List<SocketUserModel> list) {
        BasePopupView basePopupView = this.loadPopupView;
        if (basePopupView != null) {
            basePopupView.delayDismiss(300L);
        }
        this.mList.clear();
        this.mList.addAll(list);
        if (this.mAdapter == null) {
            LineUserAdapter lineUserAdapter = new LineUserAdapter(this.mContext, this.mList, this.mHallUserFlag);
            this.mAdapter = lineUserAdapter;
            lineUserAdapter.setOnItemClickListener(this.onItemClickListener);
            this.mRecyclerUser.setAdapter(this.mAdapter);
        }
        this.mAdapter.notifyDataSetChanged();
    }
}
