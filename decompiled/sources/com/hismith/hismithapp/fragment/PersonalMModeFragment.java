package com.hismith.hismithapp.fragment;

import android.content.Intent;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.PlayModeAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.fragment.PersonalMTopicFragment;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.PlayModeListModel;
import com.hismith.hismithapp.model.PlayModeModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnLoadMoreListener;
import io.reactivex.Observable;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class PersonalMModeFragment extends Fragment {
    private BaseActivity baseActivity;
    private RetrofitRequest baseRequest;
    private PlayModeAdapter mAdapter;
    private LinearLayoutManager mLayoutManager;
    private List<PlayModeModel> mList;
    private PersonalMTopicFragment.OnItemCountChangeListener mListener;
    private RecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private String mRequestId;
    private String mRequestType;
    private String userId;

    public interface OnItemCountChangeListener {
        void getItemCount(int i);
    }

    public PersonalMModeFragment() {
        this.mRequestType = "my";
    }

    public PersonalMModeFragment(String str) {
        this.userId = str;
        this.mRequestType = "userMy";
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        this.mRefreshLayout.setEnableRefresh(false);
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.baseActivity);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mList = new ArrayList();
        PlayModeAdapter playModeAdapter = new PlayModeAdapter(this.baseActivity, this.mList, this.mRequestType, true);
        this.mAdapter = playModeAdapter;
        this.mRecyclerView.setAdapter(playModeAdapter);
        this.mRequestId = "";
        playModeList(this.mRequestType, 1, "");
    }

    private void initView(View view) {
        this.mRefreshLayout = (SmartRefreshLayout) view.findViewById(R.id.personal_m_mode_refreshLayout);
        this.mRecyclerView = (RecyclerView) view.findViewById(R.id.personal_m_mode_recycler_view);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void playModeList(String str, int i, final String str2) {
        if ("userMy".equals(str)) {
            str = "my";
        }
        String str3 = str;
        BaseActivity baseActivity = this.baseActivity;
        if (baseActivity == null) {
            return;
        }
        Observable<PlayModeListModel> observeOn = this.baseRequest.getApi().playModeList((String) SharedPreferencesUtil.get(baseActivity, Constans.Token), str3, this.userId, MyApplication.getApplication().getDeviceCode(), i, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity2 = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<PlayModeListModel>(baseActivity2, baseActivity2.progressDialog, false) { // from class: com.hismith.hismithapp.fragment.PersonalMModeFragment.2
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                PersonalMModeFragment.this.mRefreshLayout.finishLoadMore();
                if (PersonalMModeFragment.this.mAdapter != null) {
                    PersonalMModeFragment.this.mAdapter.dissMorePopup();
                }
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                PersonalMModeFragment.this.mRefreshLayout.finishLoadMore();
                if (PersonalMModeFragment.this.mAdapter != null) {
                    PersonalMModeFragment.this.mAdapter.dissMorePopup();
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(PlayModeListModel playModeListModel) {
                if (playModeListModel.getSuccess() == 1) {
                    if (TextUtils.isEmpty(str2)) {
                        PersonalMModeFragment.this.mList.clear();
                        PersonalMModeFragment.this.mAdapter.notifyDataSetChanged();
                    }
                    PersonalMModeFragment.this.mRequestId = playModeListModel.getRequestId();
                    PersonalMModeFragment.this.mList.addAll(Arrays.asList(playModeListModel.getPlayModes()));
                    if (PersonalMModeFragment.this.mListener != null) {
                        PersonalMModeFragment.this.mListener.getItemCount(playModeListModel.getTotalCount());
                    }
                    PersonalMModeFragment.this.mAdapter.notifyDataSetChanged();
                }
            }
        });
    }

    private void setListener() {
        this.mRefreshLayout.setOnLoadMoreListener(new OnLoadMoreListener() { // from class: com.hismith.hismithapp.fragment.PersonalMModeFragment.1
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                int size = PersonalMModeFragment.this.mList.size();
                PersonalMModeFragment personalMModeFragment = PersonalMModeFragment.this;
                if (size > 0) {
                    personalMModeFragment.playModeList(personalMModeFragment.mRequestType, 0, PersonalMModeFragment.this.mRequestId);
                    return;
                }
                personalMModeFragment.mRequestId = "";
                PersonalMModeFragment personalMModeFragment2 = PersonalMModeFragment.this;
                personalMModeFragment2.playModeList(personalMModeFragment2.mRequestType, 1, PersonalMModeFragment.this.mRequestId);
            }
        });
    }

    @Override // androidx.fragment.app.Fragment
    public void onActivityResult(int i, int i2, Intent intent) {
        super.onActivityResult(i, i2, intent);
        LogUtil.e("创建模式界面返回requestCode：" + i + ", resultCode:" + i2);
        if (i == 17 && i2 == 17) {
            this.mRefreshLayout.autoRefresh();
        }
    }

    @Override // androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_personal_m_mode, (ViewGroup) null);
        initView(inflate);
        init();
        setListener();
        return inflate;
    }

    public void onRefresh() {
        this.mRequestId = "";
        playModeList(this.mRequestType, 1, "");
    }

    public void refreshData() {
        SmartRefreshLayout smartRefreshLayout = this.mRefreshLayout;
        if (smartRefreshLayout != null) {
            smartRefreshLayout.autoRefresh();
        }
    }

    public void setmListener(PersonalMTopicFragment.OnItemCountChangeListener onItemCountChangeListener) {
        this.mListener = onItemCountChangeListener;
    }
}
