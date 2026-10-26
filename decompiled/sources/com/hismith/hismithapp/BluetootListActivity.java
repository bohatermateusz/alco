package com.hismith.hismithapp;

import android.os.Bundle;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.fragment.BluetoothMultitudeFragment;

/* loaded from: classes3.dex */
public class BluetootListActivity extends BaseActivity implements BleManager.OnConnectStateListener, BluetoothMultitudeFragment.FragmentInteractionListener {
    private BluetoothMultitudeFragment mMultitudeFragment;

    private void init() {
    }

    private void initView() {
        this.mMultitudeFragment = (BluetoothMultitudeFragment) getSupportFragmentManager().findFragmentById(R.id.fragment_multitude);
    }

    private void setListener() {
        BleManager.getInstance(this).setOnConnectStateListener(this);
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onConnected(final int i) {
        runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.BluetootListActivity.1
            @Override // java.lang.Runnable
            public void run() {
                if (BluetootListActivity.this.mMultitudeFragment != null) {
                    BluetootListActivity.this.mMultitudeFragment.onConnected(i);
                }
            }
        });
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_bluetooth_fragment_list);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onDisconnected(final int i) {
        runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.BluetootListActivity.2
            @Override // java.lang.Runnable
            public void run() {
                if (BluetootListActivity.this.mMultitudeFragment != null) {
                    BluetootListActivity.this.mMultitudeFragment.onDisconnected(i);
                }
            }
        });
    }

    @Override // com.hismith.hismithapp.fragment.BluetoothMultitudeFragment.FragmentInteractionListener
    public void onFragmentRemoved(String str) {
        finish();
    }
}
