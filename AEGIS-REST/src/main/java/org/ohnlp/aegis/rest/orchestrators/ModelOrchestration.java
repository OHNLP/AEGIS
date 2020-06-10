package org.ohnlp.aegis.rest.orchestrators;

import org.ohnlp.aegis.api.AEGISModel;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Handles management and orchestration of {@link AEGISModel} instance
 */
public class ModelOrchestration {

    // A global lock on any read/retrieve operations
    private final Object GLOBAL_READ_LCK = new Object();

    private final ConcurrentHashMap<String, File> modelPaths;
    private final ConcurrentHashMap<String, AEGISModel> managedModels;

    public ModelOrchestration() {
        this.modelPaths = new ConcurrentHashMap<>();
        this.managedModels = new ConcurrentHashMap<>();
    }

    /**
     * Loads a saved model
     *
     * @param id         A model ID to reference the model with internally
     * @param modelClazz The model class
     * @param modelDir   The model directory
     */
    public final void load(String id, Class<? extends AEGISModel> modelClazz, File modelDir) throws IOException {
        // Lock all reads while model is loaded
        synchronized (GLOBAL_READ_LCK) {
            try {
                Constructor<? extends AEGISModel> ctor = modelClazz.getConstructor();
                AEGISModel model = ctor.newInstance();
                model.deserialize(modelDir);
            } catch (Throwable t) {
                throw new IOException("Error Loading Model", t);
            }
        }
    }

    /**
     * Retrieves a model instance. Will block if a model is currently being loaded
     *
     * @param id The model ID to retrieve
     * @return The model instance, or null if not exists
     */
    public final AEGISModel getModel(String id) {
        synchronized (GLOBAL_READ_LCK) {
            return managedModels.get(id);
        }
    }

    /**
     * Generates a copy of a currently loaded model (e.g. for training without overwriting existing production models)
     * @param id The id of the model to copy
     * @return A new instance of a model loaded
     * @throws IOException
     */
    public final AEGISModel getModelCopy(String id) throws IOException {
        synchronized (GLOBAL_READ_LCK) {
            AEGISModel base = managedModels.get(id);
            if (base == null) {
                return null;
            } else {
                Class<? extends AEGISModel> clazz = base.getClass();
                try {
                    AEGISModel model = clazz.getDeclaredConstructor().newInstance();
                    model.deserialize(modelPaths.get(id));
                    return model;
                } catch (InstantiationException | InvocationTargetException | NoSuchMethodException | IllegalAccessException e) {
                    throw new IOException("Error instantiating new model instance", e);
                }
            }
        }
    }

    /**
     * Overwrites a model instance to the appropriate ID.
     *
     * @param id The model ID to write
     * @param model The model instance
     *
     * @throws IOException
     */
    public final void writeModel(String id, AEGISModel model) throws IOException {
        synchronized (GLOBAL_READ_LCK) {
            model.serialize(modelPaths.get(id));
            managedModels.put(id, model);
        }
    }

    /**
     * Writes a model to a new ID
     * @param id    The new ID to use
     * @param path  The path to serialize the model to
     * @param model The model to write
     * @throws IOException If an exception occurs during model serialization
     * @throws IllegalArgumentException if id corresponds to an already existing and managed model (use {@link #writeModel(String, AEGISModel)} instead)
     */
    public final void writeNewModel(String id, File path, AEGISModel model) throws IOException {
        synchronized (GLOBAL_READ_LCK) {
            if (managedModels.containsKey(id)) {
                throw new IllegalArgumentException(id + " is an already managed model");
            } else {
                model.serialize(path);
                managedModels.put(id, model);
                modelPaths.put(id, path);
            }
        }

    }

}
