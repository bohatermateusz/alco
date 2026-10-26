package com.hismith.hismithapp.manager;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes3.dex */
public interface AudioChunk {

    public static abstract class AbstractAudioChunk implements AudioChunk {
        private static final double REFERENCE = 0.6d;

        @Override // com.hismith.hismithapp.manager.AudioChunk
        public double maxAmplitude() {
            short s = 0;
            for (short s2 : toShorts()) {
                if (s2 > s) {
                    s = s2;
                }
            }
            if (s > 0) {
                return Math.abs(Math.log10(s / REFERENCE) * 20.0d);
            }
            return 0.0d;
        }
    }

    public static class Bytes extends AbstractAudioChunk {
        private final byte[] bytes;

        Bytes(byte[] bArr) {
            this.bytes = bArr;
        }

        @Override // com.hismith.hismithapp.manager.AudioChunk
        public byte[] toBytes() {
            return this.bytes;
        }

        @Override // com.hismith.hismithapp.manager.AudioChunk
        public short[] toShorts() {
            byte[] bArr = this.bytes;
            short[] sArr = new short[bArr.length / 2];
            ByteBuffer.wrap(bArr).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(sArr);
            return sArr;
        }
    }

    public static class Shorts extends AbstractAudioChunk {
        private static final short SILENCE_THRESHOLD = 2700;
        private final short[] shorts;

        Shorts(short[] sArr) {
            this.shorts = sArr;
        }

        boolean isOverSilence() {
            for (short s : this.shorts) {
                if (s > 2700 || s < -2700) {
                    return true;
                }
            }
            return false;
        }

        @Override // com.hismith.hismithapp.manager.AudioChunk
        public byte[] toBytes() {
            byte[] bArr = new byte[this.shorts.length * 2];
            int i = 0;
            while (true) {
                short[] sArr = this.shorts;
                if (i >= sArr.length) {
                    return bArr;
                }
                int i2 = i * 2;
                short s = sArr[i];
                bArr[i2] = (byte) (s & 255);
                bArr[i2 + 1] = (byte) ((s & 65280) >> 8);
                i++;
            }
        }

        @Override // com.hismith.hismithapp.manager.AudioChunk
        public short[] toShorts() {
            return this.shorts;
        }
    }

    double maxAmplitude();

    byte[] toBytes();

    short[] toShorts();
}
