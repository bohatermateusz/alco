package com.hismith.hismithapp;

import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.adapter.BlackAdapter;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.BlackModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshLoadMoreListener;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class BlackListActivity extends BaseActivity implements View.OnClickListener {
    private RetrofitRequest baseRequest;
    private BlackAdapter mAdapter;
    private LinearLayoutManager mLinearLayoutManager;
    private List<BlackModel.User> mList;
    private RecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private ImageView mTitleBack;
    private TextView mTitleName;
    private TextView mTxtSend;
    private int page = 1;
    private int totalPage = 1;

    static /* synthetic */ int access$008(BlackListActivity blackListActivity) {
        int i = blackListActivity.page;
        blackListActivity.page = i + 1;
        return i;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getUserBlackList() {
        this.baseRequest.getApi().getUserBlackList((String) SharedPreferencesUtil.get(this, Constans.Token), Integer.valueOf(this.page)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<BlackModel>(this, this.progressDialog, false) { // from class: com.hismith.hismithapp.BlackListActivity.3
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                BlackListActivity.this.mRefreshLayout.finishRefresh();
                BlackListActivity.this.mRefreshLayout.finishLoadMore();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                BlackListActivity.this.mRefreshLayout.finishRefresh();
                BlackListActivity.this.mRefreshLayout.finishLoadMore();
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(BlackModel blackModel) {
                if (blackModel.getSuccess() == 1) {
                    BlackListActivity.this.totalPage = blackModel.getTotalPage();
                    if (BlackListActivity.this.page <= 1) {
                        BlackListActivity.this.mList.clear();
                    }
                    BlackListActivity.this.mList.addAll(Arrays.asList(blackModel.getUsers()));
                    BlackListActivity.this.mAdapter.notifyDataSetChanged();
                }
            }
        });
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        this.mLinearLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mList = new ArrayList();
        BlackAdapter blackAdapter = new BlackAdapter(this, this.mList);
        this.mAdapter = blackAdapter;
        this.mRecyclerView.setAdapter(blackAdapter);
        this.mRefreshLayout.autoRefresh();
    }

    private void initView() {
        this.mTitleBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleName = (TextView) findViewById(R.id.title_txt_name);
        this.mRefreshLayout = (SmartRefreshLayout) findViewById(R.id.black_refreshLayout);
        this.mRecyclerView = (RecyclerView) findViewById(R.id.black_recycler);
        this.mTxtSend = (TextView) findViewById(R.id.black_txt_send);
        LanguageUtil.setText(this, this.mTitleName, "blocking");
        LanguageUtil.setText(this, this.mTxtSend, "removeFromBlocklist");
        this.mTxtSend.setEnabled(false);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void removeUserBlackList(String str, final String str2) {
        this.baseRequest.getApi().removeUserBlackList((String) SharedPreferencesUtil.get(this, Constans.Token), str, 1).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog, true) { // from class: com.hismith.hismithapp.BlackListActivity.4
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    if (!TextUtils.isEmpty(str2)) {
                        for (String str3 : str2.split(",")) {
                            try {
                                int intValue = Integer.valueOf(str3).intValue();
                                BlackListActivity.this.mList.remove(intValue);
                                BlackListActivity.this.mAdapter.notifyItemRemoved(intValue);
                                BlackListActivity.this.mAdapter.notifyItemRangeChanged(intValue, BlackListActivity.this.mList.size());
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }
                    }
                    BlackListActivity.this.mTxtSend.setEnabled(false);
                }
            }
        });
    }

    private void setListener() {
        this.mTitleBack.setOnClickListener(this);
        this.mTxtSend.setOnClickListener(this);
        this.mRefreshLayout.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.BlackListActivity.1
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                int i = BlackListActivity.this.page;
                int i2 = BlackListActivity.this.totalPage;
                BlackListActivity blackListActivity = BlackListActivity.this;
                if (i >= i2) {
                    blackListActivity.mRefreshLayout.finishLoadMore(300);
                } else {
                    BlackListActivity.access$008(blackListActivity);
                    BlackListActivity.this.getUserBlackList();
                }
            }

            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                BlackListActivity.this.page = 1;
                BlackListActivity.this.getUserBlackList();
            }
        });
        this.mAdapter.setOnSelectedChangeListener(new BlackAdapter.OnSelectedChangeListener() { // from class: com.hismith.hismithapp.BlackListActivity.2
            @Override // com.hismith.hismithapp.adapter.BlackAdapter.OnSelectedChangeListener
            public void onSelectedChange(int i) {
                BlackListActivity blackListActivity = BlackListActivity.this;
                if (i > 0) {
                    blackListActivity.mTxtSend.setEnabled(true);
                } else {
                    blackListActivity.mTxtSend.setEnabled(false);
                }
            }
        });
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
            return;
        }
        if (id == R.id.black_txt_send) {
            final StringBuilder sb = new StringBuilder();
            final StringBuilder sb2 = new StringBuilder();
            for (int size = this.mList.size() - 1; size >= 0; size--) {
                if (this.mList.get(size).getChecked() == 1) {
                    if (sb.length() != 0) {
                        sb.append(",");
                    }
                    sb.append(this.mList.get(size).getUserId());
                    if (sb2.length() != 0) {
                        sb2.append(",");
                    }
                    sb2.append(size);
                }
            }
            if (sb.length() > 0) {
                DialogUtil.showConfirmDialog(this, "prompt", "sureRemoveBlock", null, null, new View.OnClickListener() { // from class: com.hismith.hismithapp.BlackListActivity.5
                    @Override // android.view.View.OnClickListener
                    public void onClick(View view2) {
                        LogUtil.e("移出名单：" + sb.toString() + ", 下标：" + sb2.toString());
                        BlackListActivity.this.removeUserBlackList(sb.toString(), sb2.toString());
                        DialogUtil.dismissDialog();
                    }
                });
            }
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_black_list);
        initView();
        init();
        setListener();
    }
}
