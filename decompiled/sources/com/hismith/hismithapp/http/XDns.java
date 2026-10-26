package com.hismith.hismithapp.http;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import okhttp3.Dns;

/* loaded from: classes3.dex */
public class XDns implements Dns {
    private long timeout;

    public XDns(long j) {
        this.timeout = j;
    }

    @Override // okhttp3.Dns
    public List<InetAddress> lookup(final String str) throws UnknownHostException {
        if (str == null) {
            throw new UnknownHostException("hostname == null");
        }
        try {
            FutureTask futureTask = new FutureTask(new Callable<List<InetAddress>>() { // from class: com.hismith.hismithapp.http.XDns.1
                @Override // java.util.concurrent.Callable
                public List<InetAddress> call() throws Exception {
                    return Arrays.asList(InetAddress.getAllByName(str));
                }
            });
            new Thread(futureTask).start();
            return (List) futureTask.get(this.timeout, TimeUnit.MILLISECONDS);
        } catch (Exception e) {
            UnknownHostException unknownHostException = new UnknownHostException("Broken system behaviour for dns lookup of " + str);
            unknownHostException.initCause(e);
            throw unknownHostException;
        }
    }
}
