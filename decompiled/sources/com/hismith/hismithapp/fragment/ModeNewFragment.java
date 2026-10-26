package com.hismith.hismithapp.fragment;

import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.PlayModeAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.PlayModeListModel;
import com.hismith.hismithapp.model.PlayModeModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshLoadMoreListener;
import io.reactivex.Observable;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class ModeNewFragment extends LazyLoadFragment {
    private BaseActivity baseActivity;
    private RetrofitRequest baseRequest;
    private PlayModeAdapter mAdapter;
    private LinearLayoutManager mLayoutManager;
    private List<PlayModeModel> mList;
    private RecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private String mRequestId;
    private String mRequestType = "new";
    private int mFilterOnlyThisProduct = 0;
    private String mFilterModeName = null;
    private String mFilterModeTime = null;
    private int mFilterModeSpeed = 100;
    private String mFilterModePlayCount = null;
    private String mFilterPublishDate = null;
    private String mSortModeTime = null;
    private String mSortPlayCount = null;

    private void init() {
        this.baseRequest = new RetrofitRequest();
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.baseActivity);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mList = new ArrayList();
        PlayModeAdapter playModeAdapter = new PlayModeAdapter(this.baseActivity, this.mList, this.mRequestType);
        this.mAdapter = playModeAdapter;
        this.mRecyclerView.setAdapter(playModeAdapter);
        this.mRefreshLayout.autoRefresh();
    }

    private void initView(View view) {
        this.mRefreshLayout = (SmartRefreshLayout) view.findViewById(R.id.mode_top_refreshLayout);
        this.mRecyclerView = (RecyclerView) view.findViewById(R.id.mode_top_recycler_view);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void playModeList(String str, int i, final String str2) {
        Observable<PlayModeListModel> observeOn = this.baseRequest.getApi().playModeList((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), str, MyApplication.getApplication().getDeviceCode(), i, str2, this.mSortModeTime, this.mSortPlayCount, this.mFilterOnlyThisProduct, this.mFilterModeName, this.mFilterModeTime, this.mFilterModeSpeed, this.mFilterModePlayCount, this.mFilterPublishDate).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<PlayModeListModel>(baseActivity, baseActivity.progressDialog, false) { // from class: com.hismith.hismithapp.fragment.ModeNewFragment.2
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                ModeNewFragment.this.mRefreshLayout.finishRefresh();
                ModeNewFragment.this.mRefreshLayout.finishLoadMore();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                ModeNewFragment.this.mRefreshLayout.finishRefresh();
                ModeNewFragment.this.mRefreshLayout.finishLoadMore();
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(PlayModeListModel playModeListModel) {
                if (playModeListModel.getSuccess() == 1) {
                    if (TextUtils.isEmpty(str2)) {
                        ModeNewFragment.this.mList.clear();
                        ModeNewFragment.this.mAdapter.notifyDataSetChanged();
                    }
                    ModeNewFragment.this.mRequestId = playModeListModel.getRequestId();
                    ModeNewFragment.this.mList.addAll(Arrays.asList(playModeListModel.getPlayModes()));
                    ModeNewFragment.this.mAdapter.notifyDataSetChanged();
                }
            }
        });
    }

    private void setListener() {
        this.mRefreshLayout.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.fragment.ModeNewFragment.1
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                ModeNewFragment modeNewFragment = ModeNewFragment.this;
                modeNewFragment.playModeList(modeNewFragment.mRequestType, 0, ModeNewFragment.this.mRequestId);
            }

            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                ModeNewFragment.this.mRequestId = "";
                ModeNewFragment modeNewFragment = ModeNewFragment.this;
                modeNewFragment.playModeList(modeNewFragment.mRequestType, 1, ModeNewFragment.this.mRequestId);
            }
        });
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_mode_top, (ViewGroup) null);
        initView(inflate);
        return inflate;
    }

    public void onRefresh(int i, String str, String str2, int i2, String str3, String str4, String str5, String str6) {
        SmartRefreshLayout smartRefreshLayout;
        this.mSortModeTime = str5;
        this.mSortPlayCount = str6;
        this.mFilterOnlyThisProduct = i;
        this.mFilterModeName = str;
        this.mFilterModeTime = str2;
        this.mFilterModeSpeed = i2;
        this.mFilterModePlayCount = str3;
        this.mFilterPublishDate = str4;
        if (!this.isLoad || (smartRefreshLayout = this.mRefreshLayout) == null) {
            return;
        }
        smartRefreshLayout.autoRefresh();
    }

    public void onRefresh(String str) {
        SmartRefreshLayout smartRefreshLayout;
        this.mFilterModeName = str;
        if (!this.isLoad || (smartRefreshLayout = this.mRefreshLayout) == null) {
            return;
        }
        smartRefreshLayout.autoRefresh();
    }
}
