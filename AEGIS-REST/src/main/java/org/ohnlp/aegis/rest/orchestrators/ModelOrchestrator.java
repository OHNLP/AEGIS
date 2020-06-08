package org.ohnlp.aegis.rest.orchestrators;

import org.ohnlp.aegis.api.AEGISModel;

import java.io.File;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Handles management and orchestration of {@link AEGISModel} instance
 */
public class ModelOrchestrator {

    private final ConcurrentHashMap<String, File> modelPaths;
    private final ConcurrentHashMap<String, ? extends AEGISModel> managedModels;

    public ModelOrchestrator() {
        this.modelPaths = new ConcurrentHashMap<>();
        this.managedModels = new ConcurrentHashMap<>();
    }



}
