using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UIElements;
using static ClothDynamics.GPUClothDynamicsV2;

namespace ClothDynamics
{
    [System.Serializable]
    [DefaultExecutionOrder(15200)] //When using Final IK
    public class ClothSolverGPU
    {
        public void Initialize(GPUClothDynamicsV2 dynamics)
        {
            _dynamics = dynamics;
            _collisionMeshes = _dynamics._collisionMeshes; //;m_dynamics.GetComponent<CollisionMeshesGPU>();
            _collisionMeshes._useTrisMesh = _trisMode;
            _collisionMeshes.Init(dynamics);
            Init();
        }

        void Init()
        {
            //_cs = Resources.Load<ComputeShader>("Shaders/Compute/V2/ClothSolverGPU");
            _cs = GraphicsUtilities.LoadComputeShaderAt("Shaders/Compute/V2/ClothSolverGPU");

            for (int i = 0; i < _dynamics._extensions.Length; i++)
            {
                _dynamics._extensions[i].Init(this);
            }

            _simParams.numParticles = 0;

            if (!_manualSetup) _unityColliders = GPUClothDynamicsV2.FindObjectsOfType<Collider>();

            if (_unityColliders == null) _unityColliders = new Collider[1];
            var setToRemove = new HashSet<GameObject>(_ignoreObjects);
            var colliders = _unityColliders.ToList();
            colliders.RemoveAll(x => setToRemove.Contains(x.gameObject));
            _unityColliders = colliders.ToArray();

            var abs = _absColliders.ToList();
            abs.RemoveAll(x => setToRemove.Contains(x));
            _absColliders = abs.ToArray();

            Extensions.CleanupList(ref _unityColliders);
            _unityColliders = _unityColliders.Where(x => !x.GetComponent<GPUMesh>() && !x.GetComponent<GPUSkinning>()).ToArray();

            Extensions.CleanupList(ref _absColliders);
            _absColliders = _absColliders.Where(x => !x.GetComponent<GPUMesh>() && !x.GetComponent<GPUSkinning>()).ToArray();

            Extensions.CleanupList(ref _dynamics._extensions);

        }

        public void FixedUpdate()
        {
            if (_updateMode == UpdateModes.FixedUpdate || Time.timeSinceLevelLoad < 2)
                SolverUpdate();
        }
        public void Update()
        {
            if (_useMouseGrabber)
            {
                if (!_runSim || _simParams.numParticles <= 0 || _positions == null)
                {
                    return;
                }
                if (_mouseGrabber == null)
                {
                    _mouseGrabber = new MouseGrabberGPU();
                    _mouseGrabber.Initialize(this, _dynamics, _positions, _velocities, _invMasses);
                    Debug.Log("m_mouseGrabber is Initialized");
                }
                _mouseGrabber?.HandleMouseInteraction();
            }
            if (_updateMode == UpdateModes.Update && Time.timeSinceLevelLoad >= 2)
                SolverUpdate();
        }
        public void LateUpdate()
        {
            if (_updateMode == UpdateModes.LateUpdate && Time.timeSinceLevelLoad >= 2)
                SolverUpdate();
        }
        public void OnAnimatorMove()
        {
            if (_updateMode == UpdateModes.OnAnimatorMove || Time.timeSinceLevelLoad < 2)
                SolverUpdate();
        }

        void SolverUpdate()
        {
            if (!_runSim || _simParams.numParticles <= 0 || _positions == null)
            {
                return;
            }
            if (_useMouseGrabber) _mouseGrabber?.UpdateGrappedVertex();
            UpdateColliders();
            Simulate();
            for (int n = 0; n < _dynamics._globalSimParams._sdfList.Length; n++)
            {
                var sdfTex = _dynamics._globalSimParams._sdfList[n].tex;
                if (sdfTex != null && sdfTex.sdf.GetType() == typeof(RenderTexture))
                {
                    if (sdfTex.sdfPrev != null && (sdfTex.sdfPrev.width != sdfTex.sdf.width || sdfTex.sdfPrev.height != sdfTex.sdf.height || sdfTex.sdfPrev.volumeDepth != ((RenderTexture)sdfTex.sdf).volumeDepth))
                    {
                        sdfTex.sdfPrev.Release();
                        sdfTex.sdfPrev.width = sdfTex.sdf.width;
                        sdfTex.sdfPrev.height = sdfTex.sdf.height;
                        sdfTex.sdfPrev.volumeDepth = ((RenderTexture)sdfTex.sdf).volumeDepth;
                        sdfTex.sdfPrev.Create();
                    }
                    Graphics.CopyTexture(sdfTex.sdf, sdfTex.sdfPrev);
                }

            }
        }

        public void OnDestroy()
        {
            _positions.ClearBuffer();
            //positions2.ClearBuffer();
            _normals.ClearBuffer();
            _indices.ClearBuffer();
            _velocities.ClearBuffer();
            _predicted.ClearBuffer();
            _deltas.ClearBuffer();
            _deltaCounts.ClearBuffer();
            _invMasses.ClearBuffer();
            _stretchIndices.ClearBuffer();
            _stretchLengths.ClearBuffer();

            _attachParticleIDs.ClearBuffer();
            _attachSlotIDs.ClearBuffer();
            _attachDistances.ClearBuffer();
            _attachSlotPositions.ClearBuffer();

            //sdfColliders.ClearBuffer();

            _spatialHash?.OnDestroy();

            _mouseGrabber?.OnDestroy();

            _collidableSDFsBuffer.ClearBuffer();

            if (_dummyTex != null) _dummyTex.Release();

            if (_dynamics != null)
            {
                for (int i = 0; i < _dynamics._collisionMeshes._meshObjects.Length; i++)
                {
                    Transform body = _dynamics._collisionMeshes._meshObjects[i];
                    if (body != null)
                    {
                        var meshToSDF = body.GetComponent<BodyMeshToSDF>();
                        if (meshToSDF != null)
                        {
                            if (meshToSDF._VertexBuffer != null) BodyMeshToSDF.ReleaseGraphicsBuffer(ref meshToSDF._VertexBuffer);
                            if (meshToSDF._IndexBuffer != null) BodyMeshToSDF.ReleaseGraphicsBuffer(ref meshToSDF._IndexBuffer);
                        }
                    }
                }
            }
        }

        //public SDFSpatialHashGPU _sdfSpatialHash;

        void Simulate()
        {
            float frameTime = _updateMode == UpdateModes.FixedUpdate ? Time.fixedDeltaTime : Time.deltaTime;
            float substepTime = frameTime / _dynamics._globalSimParams.numSubsteps;//_simParams.numSubsteps;

            if (_spatialHash == null)
            {
                SetSpatialHash();
            }

            _collisionMeshes.UpdateParticles(frameTime, _localSpace);

            var _invDT = 1.0f / math.max(float.Epsilon, frameTime);

            Extensions.CleanupList(ref _meshDataList);

            _listOfClothSkinners.Clear();
            int skinnersCount = 0;
            int count = _meshDataList.Count;
            for (int i = 0; i < count; i++)
            {
                var data = _meshDataList[i];
                if (data.go != null)
                {
                    var clothSkinning = data.go.GetComponent<ClothSkinningGPU>();
                    if (clothSkinning != null)
                    {
                        clothSkinning.prevNumParticles = data.prevNumParticles;
                        clothSkinning.newParticles = data.newParticles;
                        _listOfClothSkinners.Add(clothSkinning);
                        skinnersCount++;
                    }
                }
            }

            for (int i = 0; i < _dynamics._collisionMeshes._meshObjects.Length; i++)
            {
                Transform body = _dynamics._collisionMeshes._meshObjects[i];
                if (body != null)
                {
                    var meshToSDF = body.GetComponent<BodyMeshToSDF>();
#if HAS_PACKAGE_DEMOTEAM_MESHTOSDF
                    if (meshToSDF != null && meshToSDF.updateMode == MeshToSDF.UpdateMode.Explicit)
#else
                    if (meshToSDF != null && meshToSDF.updateMode == BodyMeshToSDF.UpdateMode.Explicit)
#endif
                    {
                        if (meshToSDF._CommandBuffer == null)
                            meshToSDF._CommandBuffer = new CommandBuffer() { name = "MeshToSDF" };
                        else
                            meshToSDF._CommandBuffer.Clear();

                        meshToSDF.UpdateSDF(meshToSDF._CommandBuffer);
                        Graphics.ExecuteCommandBuffer(meshToSDF._CommandBuffer);
                    }
                }
            }

            var wind = _dynamics._globalSimParams._wind;
            Vector3 windVec = Vector3.zero;
            if (wind != null)
            {
                //Don't know what the Unity WindZone algorithm does, so we're guessing here, free to edit:
                windVec = wind.transform.forward * _dynamics._globalSimParams._windIntensity * wind.windMain * 0.3333f * (1 + (UnityEngine.Random.value * wind.windTurbulence) + GetWindTurbulence(Time.fixedTime, wind.windPulseFrequency, wind.windPulseMagnitude));
                _cs.SetVector("_windVec", windVec);
            }

            //for (int i = 0; i < skinnersCount; i++)
            //{
            //    var clothSkinning = _listOfClothSkinners[i];
            //    if (clothSkinning._bodyParent != null)
            //    {
            //        if (!_localSpace)
            //            ConvertToLocalSpace(positions, clothSkinning._bodyParent.localToWorldMatrix, clothSkinning.prevNumParticles, clothSkinning.newParticles);
            //    }
            //}
            //_sdfSpatialHash.SetSpatialHash(positions);
            //_sdfSpatialHash.UpdateHash(positions);

            //// External colliders can move relatively fast, and cloth will have large velocity after colliding with them.
            //// This can produce unstable behavior, such as vertex flashing between two sides.
            //// We include a pre-stabilization step to mitigate this issue. Collision here will not influence velocity.
            CollideSDF(_positions, _positions, frameTime);
            //CollideHashSDF(positions, positions, frameTime);

            //if (_useExtraParticleCollisionStep && m_dynamics._globalSimParams.enableSelfCollision)
            //{
            //m_spatialHash.Hash(positions, _collisionMeshes._sphereDataBuffer);
            //CollideParticles(deltas, deltaCounts, positions, invMasses, m_spatialHash.neighbors, positions, frameTime);
            //ApplyDeltas(positions, deltas, deltaCounts);
            //}
            for (int substep = 0; substep < _dynamics._globalSimParams.numSubsteps; substep++)
            {
                Damping();

                if (_localSpace)
                {
                    int skinners = 0;
                    for (int i = 0; i < count; i++)
                    {
                        var data = _meshDataList[i];
                        if (data.go != null)
                        {
                            var clothSkinning = data.go.GetComponent<ClothSkinningGPU>();
                            if (clothSkinning != null)
                            {
                                Transform body = data.go.transform.parent;
                                if (body != null)
                                {
                                    if (_bodyTdsCount < skinnersCount) InitializeDiffFrames(body.position, body.lossyScale, body.rotation);
                                    UpdateDiffFrames(body, substepTime, skinners);
                                    _cs.SetVector("_windVec", Quaternion.Inverse(body.rotation) * windVec);
                                }
                                skinners++;
                            }
                            else
                            {
                                _cs.SetVector("_posVel", Vector4.zero);
                                _cs.SetVector("_rotVel", Vector4.zero);
                                _cs.SetVector("_windVec", windVec);
                            }
                        }
                        else
                        {
                            _cs.SetVector("_posVel", Vector4.zero);
                            _cs.SetVector("_rotVel", Vector4.zero);
                            _cs.SetVector("_windVec", windVec);
                        }
                        PredictPositionsLocal(_predicted, _velocities, _positions, data.prevNumParticles, data.newParticles, substepTime);
                    }
                }
                else
                {
                    PredictPositions(_predicted, _velocities, _positions, substepTime);
                }

                if (_dynamics._globalSimParams.enableSelfCollision)
                {
                    if (substep % _dynamics._globalSimParams.interleavedHash == 0)
                    {
                        _spatialHash.Hash(_predicted, _collisionMeshes._sphereDataBuffer);
                        //_sdfSpatialHash.UpdateHash(positions);
                    }

                    for (int i = 0; i < _dynamics._extensions.Length; i++)
                    {
                        _dynamics._extensions[i].CollideBodyParticles(_deltas, _deltaCounts, _predicted, _invMasses, _spatialHash.neighbors, _positions, substepTime);
                    }

                    CollideParticles(_deltas, _deltaCounts, _predicted, _invMasses, _spatialHash.neighbors, _positions, substepTime);
                }

                CollideSDF(_predicted, _positions, substepTime);
                //CollideHashSDF(predicted, positions, substepTime);

                for (int iteration = 0; iteration < _dynamics._globalSimParams.numIterations; iteration++)
                {
                    SolveStretch(_predicted, _deltas, _deltaCounts, _stretchIndices, _stretchLengths, _invMasses, _stretchLengths == null ? 0 : _stretchLengths.count);
                    SolveAttachment(_predicted, _deltas, _deltaCounts, _invMasses, _attachParticleIDs, _attachSlotIDs, _attachSlotPositions, _attachDistances, _attachParticleIDsData.Count);
                    //SolveBending(predicted, deltas, deltaCounts, bendIndices, bendAngles, invMasses, (uint)bendAngles.size(), substepTime);
                    ApplyDeltas(_predicted, _deltas, _deltaCounts);
                }

                for (int i = 0; i < skinnersCount; i++)
                {
                    var clothSkinning = _listOfClothSkinners[i];
                    if (clothSkinning != null && clothSkinning._useSurfacePush && clothSkinning.isActiveAndEnabled) clothSkinning.SurfacePush(this, _cs, _positions, _predicted, clothSkinning.prevNumParticles, clothSkinning.newParticles, BLOCK_SIZE, _localSpace);
                }

                Finalize(_velocities, _positions, _predicted, substepTime);
            }

            for (int i = 0; i < skinnersCount; i++)
            {
                var clothSkinning = _listOfClothSkinners[i];
                if (clothSkinning != null && clothSkinning.isActiveAndEnabled) clothSkinning.SkinUpdate(_cs, _positions, _velocities, clothSkinning.prevNumParticles, clothSkinning.newParticles, BLOCK_SIZE, _localSpace);
                //if (clothSkinning._bodyParent != null)
                //{
                //    if (!_localSpace)
                //        ConvertToLocalSpace(positions, clothSkinning._bodyParent.worldToLocalMatrix, clothSkinning.prevNumParticles, clothSkinning.newParticles);
                //}
            }

            ComputeNormal(_deltas, _normals, /*_localSetup ? positions2 : */_positions, _indices, _indices.count / 3);

            for (int i = 0; i < count; i++)
            {
                var data = _meshDataList[i];
                if (data.go != null)
                {
                    var clothObject = data.go.GetComponent<ClothObjectGPU>(); //TODO
                    if (clothObject != null) clothObject.ComputeSkinningHD();
                }
            }
        }

        private void Damping()
        {
            // damp velocity
            //if (m_dynamics._globalSimParams._dampingMethod == DampingMethod.smartDamping || m_dynamics._globalSimParams._dampingMethod == DampingMethod.smartAndSimpleDamping)
            //{
            //    SmartDamping();
            //}
            if (_dynamics._globalSimParams._dampingMethod == GPUClothDynamicsV2.DampingMethods.simpleDamping/* || m_dynamics._globalSimParams._dampingMethod == GPUClothDynamicsV2.DampingMethods.smartAndSimpleDamping*/)
            {
                int DampVelocities_Kernel = 15;
                _cs.SetFloat("_clampVel", _dynamics._globalSimParams._clampVel);
                _cs.SetFloat("_dampingVel", _dynamics._globalSimParams._dampingVel);
                _cs.SetInt("params_numParticles", _simParams.numParticles);
                _cs.SetBuffer(DampVelocities_Kernel, "velocities", _velocities);
                _cs.Dispatch(DampVelocities_Kernel, _simParams.numParticles.GetComputeShaderThreads(BLOCK_SIZE), 1, 1);
            }
        }

        void ComputeNormal(ComputeBuffer deltas, ComputeBuffer normals, ComputeBuffer positions, ComputeBuffer indices, int numTriangles)
        {
            int ComputeTriangleNormals_Kernel = 8;
            _cs.SetInt("numTriangles", numTriangles);
            _cs.SetBuffer(ComputeTriangleNormals_Kernel, "deltas", deltas);
            //m_cs.SetBuffer(ComputeTriangleNormals_Kernel, "normals", normals);
            _cs.SetBuffer(ComputeTriangleNormals_Kernel, "positions", positions);
            _cs.SetBuffer(ComputeTriangleNormals_Kernel, "indices", indices);
            _cs.Dispatch(ComputeTriangleNormals_Kernel, numTriangles.GetComputeShaderThreads(BLOCK_SIZE), 1, 1);

            int ComputeVertexNormals_Kernel = 9;
            _cs.SetBuffer(ComputeVertexNormals_Kernel, "deltas", deltas);
            _cs.SetBuffer(ComputeVertexNormals_Kernel, "normals", normals);
            _cs.Dispatch(ComputeVertexNormals_Kernel, _simParams.numParticles.GetComputeShaderThreads(BLOCK_SIZE), 1, 1);
        }

        void ConvertToLocalSpace(ComputeBuffer positions, float4x4 modelMatrix, int start, int count)
        {
            int ConvertToLocalSpace_Kernel = 10;
            //m_cs.SetInt("params_numParticles", _simParams.numParticles);
            _cs.SetMatrix("_transformMatrix", modelMatrix);
            _cs.SetBuffer(ConvertToLocalSpace_Kernel, "positions", positions);
            _cs.SetInt("start", start);
            _cs.SetInt("count", count);
            _cs.Dispatch(ConvertToLocalSpace_Kernel, _simParams.numParticles.GetComputeShaderThreads(BLOCK_SIZE), 1, 1);
        }

        void InitializePositions(ComputeBuffer positions, int start, int count, float4x4 modelMatrix)
        {
            int InitializePositions_Kernel = 0;
            _cs.SetBuffer(InitializePositions_Kernel, "positions", positions);
            _cs.SetInt("start", start);
            _cs.SetInt("count", count);
            _cs.SetMatrix("modelMatrix", modelMatrix);
            _cs.Dispatch(InitializePositions_Kernel, count.GetComputeShaderThreads(BLOCK_SIZE), 1, 1);
        }

        void PredictPositions(ComputeBuffer predicted, ComputeBuffer velocities, ComputeBuffer positions, float deltaTime)
        {
            int PredictPositions_Kernel = 1;
            _cs.SetInt("params_numParticles", _simParams.numParticles);
            _cs.SetVector("params_gravity", _dynamics._globalSimParams.gravity);
            _cs.SetBuffer(PredictPositions_Kernel, "predicted", predicted);
            _cs.SetBuffer(PredictPositions_Kernel, "velocities", velocities);
            _cs.SetBuffer(PredictPositions_Kernel, "positions", positions);
            _cs.SetBuffer(PredictPositions_Kernel, "invMasses", _invMasses);
            _cs.SetFloat("deltaTime", deltaTime);
            _cs.Dispatch(PredictPositions_Kernel, _simParams.numParticles.GetComputeShaderThreads(BLOCK_SIZE), 1, 1);
        }

        void PredictPositionsLocal(ComputeBuffer predicted, ComputeBuffer velocities, ComputeBuffer positions, int start, int count, float deltaTime)
        {
            int PredictPositions_Kernel = 14;
            _cs.SetInt("params_numParticles", _simParams.numParticles);
            _cs.SetVector("params_gravity", _dynamics._globalSimParams.gravity);
            _cs.SetBuffer(PredictPositions_Kernel, "predicted", predicted);
            _cs.SetBuffer(PredictPositions_Kernel, "velocities", velocities);
            _cs.SetBuffer(PredictPositions_Kernel, "positions", positions);
            _cs.SetBuffer(PredictPositions_Kernel, "invMasses", _invMasses);
            _cs.SetInt("start", start);
            _cs.SetInt("count", count);
            _cs.SetFloat("deltaTime", deltaTime);
            _cs.Dispatch(PredictPositions_Kernel, _simParams.numParticles.GetComputeShaderThreads(BLOCK_SIZE), 1, 1);
        }

        void SolveStretch(ComputeBuffer predicted, ComputeBuffer deltas, ComputeBuffer deltaCounts, ComputeBuffer stretchIndices, ComputeBuffer stretchLengths, ComputeBuffer invMasses, int numConstraints)
        {
            if (numConstraints == 0) return;

            int SolveStretch_Kernel = 2;
            _cs.SetBuffer(SolveStretch_Kernel, "predicted", predicted);
            _cs.SetBuffer(SolveStretch_Kernel, "deltas", deltas);
            _cs.SetBuffer(SolveStretch_Kernel, "deltaCounts", deltaCounts);
            _cs.SetBuffer(SolveStretch_Kernel, "stretchIndices", stretchIndices);
            _cs.SetBuffer(SolveStretch_Kernel, "stretchLengths", stretchLengths);
            _cs.SetBuffer(SolveStretch_Kernel, "invMasses", invMasses);
            _cs.SetInt("numConstraints", numConstraints);
            _cs.Dispatch(SolveStretch_Kernel, numConstraints.GetComputeShaderThreads(BLOCK_SIZE), 1, 1);
        }

        void SolveAttachment(ComputeBuffer predicted, ComputeBuffer deltas, ComputeBuffer deltaCounts, ComputeBuffer invMass, ComputeBuffer attachParticleIDs, ComputeBuffer attachSlotIDs, ComputeBuffer attachSlotPositions, ComputeBuffer attachDistances, int numConstraints)
        {
            if (numConstraints == 0) return;
            //Debug.Log("SolveAttachment " + numConstraints);
            UpdateAttachSlot();
            int SolveAttachment_Kernel = 3;
            _cs.SetFloat("params_longRangeStretchiness", _dynamics._globalSimParams.longRangeStretchiness);
            _cs.SetBuffer(SolveAttachment_Kernel, "predicted", predicted);
            _cs.SetBuffer(SolveAttachment_Kernel, "deltas", deltas);
            _cs.SetBuffer(SolveAttachment_Kernel, "deltaCounts", deltaCounts);
            _cs.SetBuffer(SolveAttachment_Kernel, "invMasses", invMass);
            _cs.SetBuffer(SolveAttachment_Kernel, "attachParticleIDs", attachParticleIDs);
            _cs.SetBuffer(SolveAttachment_Kernel, "attachSlotIDs", attachSlotIDs);
            _cs.SetBuffer(SolveAttachment_Kernel, "attachSlotPositions", attachSlotPositions);
            _cs.SetBuffer(SolveAttachment_Kernel, "attachDistances", attachDistances);
            _cs.SetInt("numConstraints", numConstraints);
            _cs.Dispatch(SolveAttachment_Kernel, numConstraints.GetComputeShaderThreads(1024), 1, 1);
        }

        void ApplyDeltas(ComputeBuffer predicted, ComputeBuffer deltas, ComputeBuffer deltaCounts)
        {
            int ApplyDeltas_Kernel = 4;
            _cs.SetInt("params_numParticles", _simParams.numParticles);
            _cs.SetFloat("params_relaxationFactor", _dynamics._globalSimParams.relaxationFactor);
            _cs.SetBuffer(ApplyDeltas_Kernel, "predicted", predicted);
            _cs.SetBuffer(ApplyDeltas_Kernel, "deltas", deltas);
            _cs.SetBuffer(ApplyDeltas_Kernel, "deltaCounts", deltaCounts);
            _cs.Dispatch(ApplyDeltas_Kernel, _simParams.numParticles.GetComputeShaderThreads(BLOCK_SIZE), 1, 1);
        }

        void Finalize(ComputeBuffer velocities, ComputeBuffer positions, ComputeBuffer predicted, float deltaTime)
        {
            int Finalize_Kernel = 5;
            _cs.SetInt("params_numParticles", _simParams.numParticles);
            _cs.SetFloat("params_maxSpeed", _dynamics._globalSimParams.maxSpeed);
            _cs.SetFloat("params_damping", _dynamics._globalSimParams.damping);
            _cs.SetBuffer(Finalize_Kernel, "velocities", velocities);
            _cs.SetBuffer(Finalize_Kernel, "positions", positions);
            _cs.SetBuffer(Finalize_Kernel, "predicted", predicted);
            _cs.SetFloat("deltaTime", deltaTime);
            _cs.Dispatch(Finalize_Kernel, _simParams.numParticles.GetComputeShaderThreads(BLOCK_SIZE), 1, 1);
        }

        void CollideSDF(ComputeBuffer predicted, ComputeBuffer positions, float deltaTime)
        {
            //if (numColliders == 0) return;
            int CollideSDF_Kernel = 11;// 6;
            _cs.SetInt("params_numParticles", _simParams.numParticles);
            _cs.SetFloat("params_friction", _dynamics._globalSimParams.friction);
            //m_cs.SetFloat("params_collisionMargin", m_dynamics._globalSimParams.collisionMargin);
            _cs.SetBuffer(CollideSDF_Kernel, "predicted", predicted);
            //m_cs.SetBuffer(CollideSDF_Kernel, "colliders", colliders);
            _cs.SetBuffer(CollideSDF_Kernel, "positions", positions);
            //m_cs.SetInt("numColliders", (int)numColliders);
            _cs.SetFloat("deltaTime", deltaTime);

            _cs.SetFloat("_TimeStep", Time.deltaTime);
            
            _cs.SetInt("_sdfListCount", _dynamics._globalSimParams._sdfList.Length);

            uint SDF_TEX_COUNT = 8; //Set this also in the compute shader if value is higher than 8
            var matArray = new Matrix4x4[SDF_TEX_COUNT];
            var bCenterArray = new Vector4[SDF_TEX_COUNT];
            var bSizeArray = new Vector4[SDF_TEX_COUNT];
            var scaleArray = new Vector4[SDF_TEX_COUNT];
            var gridSizeArray = new Vector4[SDF_TEX_COUNT];

            for (int n = 0; n < SDF_TEX_COUNT; n++)
            {
                if (_dynamics._globalSimParams._sdfList != null && n < _dynamics._globalSimParams._sdfList.Length && _dynamics._globalSimParams._sdfList[n] != null && _dynamics._globalSimParams._sdfList[n].tex != null)
                {
                    var sdfTex = _dynamics._globalSimParams._sdfList[n].tex;
                    var tex = (RenderTexture)sdfTex.sdf;
                    Vector3 gridSize = new Vector4(tex.width, tex.height, tex.volumeDepth);
                    Vector4 bCenter = sdfTex.transform.localPosition;
                    Vector4 bSize = sdfTex.size;
                    Vector3 scale = (Vector3)bSize / (float3)gridSize;
                    bCenter.w = _dynamics._globalSimParams._sdfList[n]._sdfOffset;
                    bSize.w = _dynamics._globalSimParams._sdfList[n]._sdfIntensity * 0.01f;

                    var mat = Matrix4x4.TRS(_localSpace ? Vector3.zero : -sdfTex.transform.parent.position, sdfTex.transform.localRotation, Vector3.one);

                    matArray[n] = mat;
                    bCenterArray[n] = bCenter;
                    bSizeArray[n] = bSize;
                    scaleArray[n] = scale;
                    gridSizeArray[n] = gridSize;
                    _cs.SetTexture(CollideSDF_Kernel, "_sdfVoxelData" + n, tex);
                    _cs.SetTexture(CollideSDF_Kernel, "_sdfVoxelDataPrev" + n, sdfTex.sdfPrev);
                }
                else
                {
                    if (_dummyTex == null) { _dummyTex = new RenderTexture(1, 1, 0, RenderTextureFormat.RHalf); _dummyTex.volumeDepth = 1; _dummyTex.dimension = UnityEngine.Rendering.TextureDimension.Tex3D; }
                    _cs.SetTexture(CollideSDF_Kernel, "_sdfVoxelData" + n, _dummyTex);
                    _cs.SetTexture(CollideSDF_Kernel, "_sdfVoxelDataPrev" + n, _dummyTex);
                    gridSizeArray[n] = Vector4.zero;
                }
            }
            _cs.SetMatrixArray("_sdfMatrixInv", matArray);
            _cs.SetVectorArray("_bCenter", bCenterArray);
            _cs.SetVectorArray("_bSize", bSizeArray);
            _cs.SetVectorArray("_scale", scaleArray);
            _cs.SetVectorArray("_gridSize", gridSizeArray);

            _cs.SetBuffer(CollideSDF_Kernel, "_collidableSDFs", _collidableSDFsBuffer);
            _cs.SetInt("_numCollidableSDFs", _numCollidableSDFs);
            _cs.SetFloat("_collidableObjectsBias", _dynamics._globalSimParams.sdfCollisionMargin);
            _cs.SetFloat("_dynamicFriction", _dynamics._globalSimParams.friction);

            _cs.Dispatch(CollideSDF_Kernel, _simParams.numParticles.GetComputeShaderThreads(BLOCK_SIZE), 1, 1);
        }

        //void CollideHashSDF(ComputeBuffer predicted, ComputeBuffer positions, float deltaTime)
        //{
        //    //if (numColliders == 0) return;
        //    int CollideSDF_Kernel = 20;
        //    _cs.SetInt("params_numParticles", _simParams.numParticles);
        //    //_cs.SetInt("_numCollidableSDFs", _numCollidableSDFs);
        //    _cs.SetFloat("_collidableObjectsBias", _dynamics._globalSimParams.sdfCollisionMargin);
        //    _cs.SetFloat("_dynamicFriction", _dynamics._globalSimParams.friction);
        //    _cs.SetFloat("params_friction", _dynamics._globalSimParams.friction);
        //    _cs.SetFloat("deltaTime", deltaTime);

        //    _cs.SetBuffer(CollideSDF_Kernel, "predicted", predicted);
        //    _cs.SetBuffer(CollideSDF_Kernel, "positions", positions);
        //    _cs.SetBuffer(CollideSDF_Kernel, "_collidableSDFs", _collidableSDFsBuffer);

        //    int numObjects = _simParams.numParticles + _sdfSpatialHash._sdfPositions.count;
        //    _cs.SetInt("params_numObjects", numObjects);
        //    _cs.SetInt("params_maxNumNeighbors", _sdfSpatialHash.maxNumNeighbors);
        //    _cs.SetInt("_sdfPositionsCount", _sdfSpatialHash._sdfPositions.count);
        //    _cs.SetBuffer(CollideSDF_Kernel, "neighborsSDF", _sdfSpatialHash.neighbors);
        //    _cs.SetBuffer(CollideSDF_Kernel, "_sdfPositions", _sdfSpatialHash._sdfPositions);
        //    //m_cs.SetBuffer(CollideParticles_Kernel, "_getNeighbours", _getNeighbours);
        //    //_cs.Dispatch(CollideSDF_Kernel, positions.count.GetComputeShaderThreads(BLOCK_SIZE), 1, 1);

        //    _cs.Dispatch(CollideSDF_Kernel, _simParams.numParticles.GetComputeShaderThreads(BLOCK_SIZE), 1, 1);
        //}



        private void CollideParticles(ComputeBuffer deltas, ComputeBuffer deltaCounts, ComputeBuffer predicted, ComputeBuffer invMasses, ComputeBuffer neighbors, ComputeBuffer positions, float deltaTime)
        {
            //ScopedTimerGPU timer("Solver_CollideParticles");
            int CollideParticles_Kernel = 7;
            _cs.SetBool("_trisMode", _trisMode);
            _cs.SetFloat("deltaTime", deltaTime);

            int meshesParticlesNum = _simParams.numParticles + (_collisionMeshes._sphereDataBuffer.count > 1 ? _collisionMeshes._sphereDataBuffer.count : 0);
            _cs.SetInt("params_numObjects", meshesParticlesNum);
            _cs.SetInt("params_numParticles", _simParams.numParticles);
            _cs.SetInt("params_maxNumNeighbors", _dynamics._globalSimParams.maxNumNeighbors);
            _cs.SetFloat("params_particleDiameter", _dynamics._globalSimParams.particleDiameter);
            _cs.SetBuffer(CollideParticles_Kernel, "deltas", deltas);
            _cs.SetBuffer(CollideParticles_Kernel, "deltaCounts", deltaCounts);
            _cs.SetBuffer(CollideParticles_Kernel, "predicted", predicted);
            _cs.SetBuffer(CollideParticles_Kernel, "invMasses", invMasses);
            _cs.SetBuffer(CollideParticles_Kernel, "neighbors", neighbors);
            _cs.SetBuffer(CollideParticles_Kernel, "positions", positions);
            _cs.SetBuffer(CollideParticles_Kernel, "_sphereDataBuffer", _collisionMeshes._sphereDataBuffer);
            //m_cs.SetBuffer(CollideParticles_Kernel, "_spherePosPredicted", _collectSpheres._spherePosPredicted);
            _cs.SetInt("_sphereDataBufferCount", _collisionMeshes._sphereDataBuffer.count);
            //m_cs.SetInt("_loopCount", _loopCount);

            _cs.Dispatch(CollideParticles_Kernel, _simParams.numParticles.GetComputeShaderThreads(BLOCK_SIZE), 1, 1);

            ApplyDeltas(predicted, deltas, deltaCounts);
        }

        public int2 AddCloth(GPUClothDynamicsV2 dynamics, GameObject go, Mesh mesh, float4x4 modelMatrix, float particleDiameter)
        {
            if (_dynamics == null) Initialize(dynamics);
            if (_dynamics._globalSimParams.particleDiameter != 0)
            {
                particleDiameter = math.min(_dynamics._globalSimParams.particleDiameter, particleDiameter);
            }

            int prevNumParticles = _simParams.numParticles;
            int newParticles = mesh.vertexCount;

            int meshId = _meshDataList.Count;
            _meshDataList.Add(new MeshData() { go = go, id = meshId, newParticles = newParticles, prevNumParticles = prevNumParticles, modelMatrix = modelMatrix });

            // Set global parameters
            _simParams.numIterations = _dynamics._globalSimParams.numIterations;
            _simParams.numSubsteps = _dynamics._globalSimParams.numSubsteps;
            _simParams.numParticles += newParticles;
            _simParams.particleDiameter = particleDiameter;
            _simParams.deltaTime = _updateMode == UpdateModes.FixedUpdate ? Time.fixedDeltaTime : Time.deltaTime;//Time.fixedDeltaTime;
            _simParams.maxSpeed = 2 * particleDiameter / _simParams.deltaTime * _simParams.numSubsteps;

            _dynamics._globalSimParams.particleDiameter = _simParams.particleDiameter; //TODO remove?
            _dynamics._globalSimParams.deltaTime = _simParams.deltaTime; //TODO remove?
            _dynamics._globalSimParams.maxSpeed = _simParams.maxSpeed; //TODO remove?
            _dynamics._globalSimParams.numParticles = _simParams.numParticles; //TODO remove?

            // Allocate managed buffers

            _positionsData.AddRange(mesh.vertices);
            _normalsData.AddRange(mesh.normals);

            //Color[] baseColors = mesh.colors; //TODO
            Color[] colorsTmp = mesh.colors;

            bool useVertexColors = true;
            if (colorsTmp == null || colorsTmp.Length != newParticles)
                useVertexColors = false;

            var velocity = new Vector4(0, 0, 0, 0);
            for (int i = 0; i < mesh.vertexCount; i++)
            {
                velocity.w = useVertexColors ? colorsTmp[i].r : 1;
                _velocitiesData.Add(velocity);
            }

            var tris = mesh.triangles;
            var indicesData = new int[tris.Length];
            for (int i = 0; i < indicesData.Length; i++)
            {
                indicesData[i] = tris[i] + prevNumParticles;
            }
            _indicesData.AddRange(indicesData);

            return new int2(prevNumParticles, newParticles);
        }

        public void UpdateBuffers()
        {
            Debug.Log("UpdateBuffers");

            _positions.ClearBuffer();
            _positions = new ComputeBuffer(_positionsData.Count, sizeof(float) * 3);
            _positions.SetData(_positionsData);
            //positions2.ClearBuffer();
            //positions2 = new ComputeBuffer(_positionsData.Count, sizeof(float) * 3);
            //positions2.SetData(_positionsData);
            _normals.ClearBuffer();
            _normals = new ComputeBuffer(_normalsData.Count, sizeof(float) * 3);
            _normals.SetData(_normalsData);
            _indices.ClearBuffer();
            _indices = new ComputeBuffer(_indicesData.Count, sizeof(int));
            _indices.SetData(_indicesData);

            _velocities.ClearBuffer();
            _velocities = new ComputeBuffer(_simParams.numParticles, sizeof(float) * 4);
            _velocities.SetData(_velocitiesData);
            _predicted.ClearBuffer();
            _predicted = new ComputeBuffer(_simParams.numParticles, sizeof(float) * 3);
            _predicted.SetData(_positionsData);
            _deltas.ClearBuffer();
            _deltas = new ComputeBuffer(_simParams.numParticles, sizeof(int) * 3);
            _deltaCounts.ClearBuffer();
            _deltaCounts = new ComputeBuffer(_simParams.numParticles, sizeof(int));
            _invMasses.ClearBuffer();
            _invMasses = new ComputeBuffer(_simParams.numParticles, sizeof(float));
            _invMasses.SetData(_invMassesData);


            //if (_meshDataList.Count > 0 && _meshDataList[0] != null)
            //    ConvertToLocalSpace(positions, velocities, _meshDataList[0].go.transform.worldToLocalMatrix);

            Extensions.CleanupList(ref _meshDataList);

            for (int i = 0; i < _meshDataList.Count; i++)
            {
                if (_meshDataList[i].go == null) continue;

                _meshDataList[i].go.GetComponent<ClothObjectGPU>().SetMatProperties();

                var prevNumParticles = _meshDataList[i].prevNumParticles;
                var modelMatrix = _meshDataList[i].modelMatrix;
                var newParticles = _meshDataList[i].newParticles;
                InitializePositions(_positions, prevNumParticles, newParticles, modelMatrix);
                //InitializePositions(positions2, prevNumParticles, newParticles, modelMatrix);
            }


            if (_spatialHash != null) _spatialHash.OnDestroy();
            SetSpatialHash();

            if (_useMouseGrabber && _mouseGrabber == null)
            {
                _mouseGrabber = new MouseGrabberGPU();
                _mouseGrabber.Initialize(this, _dynamics, _positions, _velocities, _invMasses);
                Debug.Log("m_mouseGrabber is Initialized");
            }
        }

        private void SetSpatialHash()
        {
            var meshesParticlesNum = _simParams.numParticles + (_collisionMeshes._sphereDataBuffer.count > 1 ? _collisionMeshes._sphereDataBuffer.count : 0);
            _spatialHash = new SpatialHashGPU(_dynamics, _dynamics._globalSimParams.particleDiameter, meshesParticlesNum);
            _spatialHash.SetInitialPositions(_positions, _collisionMeshes._sphereDataBuffer);
        }

        public void AddStretch(int idx1, int idx2, float distance)
        {
            _stretchIndicesData.Add(idx1);
            _stretchIndicesData.Add(idx2);
            _stretchLengthsData.Add(distance);
        }

        public void AddMasses(float value = 1)
        {
            int count = _invMassesData.Count;
            if (count < _simParams.numParticles)
            {
                for (int i = count; i < _simParams.numParticles; i++)
                {
                    //if (_invMassesData.Count < _simParams.numParticles)
                    _invMassesData.Add(value);
                }
            }
            //Debug.Log("_invMassesData " + _invMassesData.Count);
        }

        public void UpdateStretch()
        {

            if (_stretchIndices == null || _stretchIndices.count < _stretchIndicesData.Count)
            {
                _stretchIndices.ClearBuffer();
                _stretchIndices = new ComputeBuffer(math.max(1, _stretchIndicesData.Count), sizeof(int));//TODO size?
            }
            _stretchIndices.SetData(_stretchIndicesData);

            if (_stretchLengths == null || _stretchLengths.count < _stretchLengthsData.Count)
            {
                _stretchLengths.ClearBuffer();
                _stretchLengths = new ComputeBuffer(math.max(1, _stretchLengthsData.Count), sizeof(float));//TODO size?
            }
            _stretchLengths.SetData(_stretchLengthsData);
        }

        public void AddAttachSlot(float3 attachSlotPos, Transform t)
        {
            _connectedObjects.Add(new ConnectedData() { transform = t, offset = attachSlotPos - (float3)t.position });
            _attachSlotPositionsData.Add(attachSlotPos);
        }

        public void UpdateAttachSlot()
        {
            int count = _attachSlotPositionsData.Count;
            if (_attachSlotPositions == null || _attachSlotPositions.count < count)
            {
                _attachSlotPositions.ClearBuffer();
                _attachSlotPositions = new ComputeBuffer(math.max(1, count), sizeof(float) * 3, ComputeBufferType.Default, ComputeBufferMode.SubUpdates);//TODO size?
            }
            if (_attachSlotPositionsData.Count > 0)
            {
                var buffer = _attachSlotPositions.BeginWrite<float3>(0, count);
                for (int i = 0; i < count; i++)
                {
                    var obj = _connectedObjects[i];
                    _attachSlotPositionsData[i] = (float3)obj.transform.position + obj.offset;
                    buffer[i] = _attachSlotPositionsData[i];
                }
                _attachSlotPositions.EndWrite<float3>(count);
                //attachSlotPositions.SetData(attachSlotPositionsData);
            }
        }
        public void AddAttach(int particleIndex, int slotIndex, float distance)
        {
            if (distance == 0) _invMassesData[particleIndex] = 0;
            _attachParticleIDsData.Add(particleIndex);
            _attachSlotIDsData.Add(slotIndex);
            _attachDistancesData.Add(distance);
        }

        public void RemoveAllAttach()
        {
            _connectedObjects.Clear();
            _attachSlotPositionsData.Clear();
            _attachParticleIDsData.Clear();
            _attachSlotIDsData.Clear();
            _attachDistancesData.Clear();
        }
        public void UpdateAttach()
        {
            if (_attachParticleIDs == null || _attachParticleIDs.count < _attachParticleIDsData.Count)
            {
                _attachParticleIDs.ClearBuffer();
                _attachParticleIDs = new ComputeBuffer(math.max(1, _attachParticleIDsData.Count), sizeof(int));//TODO size?
            }
            if (_attachParticleIDsData.Count > 0) _attachParticleIDs.SetData(_attachParticleIDsData);

            if (_attachSlotIDs == null || _attachSlotIDs.count < _attachSlotIDsData.Count)
            {
                _attachSlotIDs.ClearBuffer();
                _attachSlotIDs = new ComputeBuffer(math.max(1, _attachSlotIDsData.Count), sizeof(int));//TODO size?
            }
            if (_attachSlotIDsData.Count > 0) _attachSlotIDs.SetData(_attachSlotIDsData);

            if (_attachDistances == null || _attachDistances.count < _attachDistancesData.Count)
            {
                _attachDistances.ClearBuffer();
                _attachDistances = new ComputeBuffer(math.max(1, _attachDistancesData.Count), sizeof(float));//TODO size?
            }
            if (_attachDistancesData.Count > 0) _attachDistances.SetData(_attachDistancesData);
        }
        //void AddBend(uint idx1, uint idx2, uint idx3, uint idx4, float angle)
        //{
        //    bendIndices.push_back(idx1);
        //    bendIndices.push_back(idx2);
        //    bendIndices.push_back(idx3);
        //    bendIndices.push_back(idx4);
        //    bendAngles.push_back(angle);
        //}

        //public Transform _bodyParent;

        void UpdateColliders()
        {
            Collider[] colliders = _unityColliders;
            int collidersCount = 0;
            int[] mapColliders = new int[colliders.Length];
            for (int i = 0; i < colliders.Length; i++)
            {
                Collider c = colliders[i];
                if (c == null || !c.enabled) continue;
                mapColliders[collidersCount] = i;
                collidersCount++;
            }

            GameObject[] collidableObjects = _absColliders;

            bool sdfsCreated = true;
            if (_sdfObjs == null || _collidableSDFsBuffer.count < (collidersCount + collidableObjects.Length))
            {
                _sdfObjs = new List<SDFObject>();
                _sdfObjs.Clear();

                for (int i = 0; i < collidersCount; i++)
                {
                    Collider c = colliders[mapColliders[i]];
                    var root = c.transform.root != c.transform ? c.transform.root : null;
                    _sdfObjs.Add(new SDFObject() { transform = c.transform, parent = root });
                }
                for (int i = 0; i < collidableObjects.Length; i++)
                {
                    GameObject c = collidableObjects[i];
                    var root = c.transform.root != c.transform ? c.transform.root : null;
                    _sdfObjs.Add(new SDFObject() { transform = c.transform, parent = root });
                }

                _numCollidableSDFs = _sdfObjs.Count;
                _collidableSDFs = new CollidableSDFStruct[_numCollidableSDFs];
                int size = System.Runtime.InteropServices.Marshal.SizeOf(typeof(CollidableSDFStruct));
                _collidableSDFsBuffer = new ComputeBuffer(Mathf.Max(1, _numCollidableSDFs), size, ComputeBufferType.Default, ComputeBufferMode.SubUpdates);
                sdfsCreated = false;
            }

            if (_numCollidableSDFs > 0)
            {
                _sdfBuffer = _collidableSDFsBuffer.BeginWrite<CollidableSDFStruct>(0, _numCollidableSDFs);
                for (int i = 0; i < _numCollidableSDFs; i++)
                {
                    SetupSDFCollider(i, _localSpace);
                    int tdNum = !sdfsCreated ? InitializeDiffFrames(_sdfObjs[i].transform.position, _sdfObjs[i].transform.lossyScale, _sdfObjs[i].transform.rotation, _sdfObjs[i].parent, _localSpace)
                        : i;
                    UpdateDiffFrames(_sdfObjs[i].transform, _sdfObjs[i].parent, Time.deltaTime, tdNum, out _collidableSDFs[i].posVel, out _collidableSDFs[i].rotVel, _localSpace);
                    _sdfBuffer[i] = _collidableSDFs[i];
                }
                _collidableSDFsBuffer.EndWrite<CollidableSDFStruct>(_numCollidableSDFs);
            }
        }
        private Vector3 SetupCapsule(Collider c, out Quaternion rot)
        {
            rot = Quaternion.identity;
            var collider = ((CapsuleCollider)c);
            var vec = new Vector3(collider.radius, (collider.height - 1) * 0.25f, collider.radius);
            if (collider.direction == 0)
            {
                rot.eulerAngles = new Vector3(0, 0, 90);
            }
            if (collider.direction == 2)
            {
                rot.eulerAngles = new Vector3(90, 0, 0);
            }
            return vec;
        }
        private void SetupSDFCollider(int i, bool localSetup = false)
        {
            int sdfType = _sdfObjs[i].transform.GetComponent<CylinderCollider>() ? 4 :
                          _sdfObjs[i].transform.GetComponent<RoundConeCollider>() ? 3 :
                          _sdfObjs[i].transform.GetComponent<BoxCollider>() ? 0 :
                          _sdfObjs[i].transform.GetComponent<CapsuleCollider>() ? 1 :
                          _sdfObjs[i].transform.GetComponent<SphereCollider>() ? 2 :
                          5;
            Collider c = sdfType == 0 ? _sdfObjs[i].transform.GetComponent<BoxCollider>() :
                        sdfType == 1 ? (Collider)_sdfObjs[i].transform.GetComponent<CapsuleCollider>() :
                        sdfType == 2 ? (Collider)_sdfObjs[i].transform.GetComponent<SphereCollider>() :
                        null;

            var lossyScale = _sdfObjs[i].transform.lossyScale;
            Quaternion extraRot = Quaternion.identity;
            Vector4 extent = sdfType == 0 ? (c == null ? Vector3.one * 0.5f : ((BoxCollider)c).size / 2f)
                        : sdfType == 1 ? (c == null ? Vector3.one * 0.5f : SetupCapsule(c, out extraRot))
                        : sdfType == 2 ? (c == null ? Vector3.one * 0.5f : Vector3.one * ((SphereCollider)c).radius * 0.5f)
                        : sdfType == 3 ? _sdfObjs[i].transform.GetComponent<RoundConeCollider>().r1r2h
                        : Vector3.one * 0.5f;
            _collidableSDFs[i].center = _sdfObjs[i].transform.position;
            if (sdfType < 3 && c != null)
            {
                _collidableSDFs[i].center += _sdfObjs[i].transform.TransformVector(sdfType == 0 ? ((BoxCollider)c).center :
                                                                                   sdfType == 1 ? (((CapsuleCollider)c).center + Vector3.up * 0.5f) :
                                                                                   sdfType == 2 ? ((SphereCollider)c).center : Vector3.zero);
            }

            //if (_debugMode && sdfType == 3) _sdfObjs[i].GetComponent<RoundConeCollider>()._showGizmos = _debugMode;

            //Debug.DrawRay(_collidableSDFs[i].center, Vector3.up * 0.1f, Color.white);

            extent = sdfType == 3 ? extent : Vector4.Scale(lossyScale, extent);
            var cfc = _sdfObjs[i].transform.GetComponent<ClothFrictionCollider>();
            extent.w = cfc ? 1.0f - cfc.friction : 1;
            if (sdfType == 5)
            {
                if (_sdfObjs[i].transform.GetComponent<MeshRenderer>())
                { extent.y = _usePlaneScaleY ? extent.y : 0.0f; sdfType = 6; }
                else
                {
                    extent *= 0.5f; //ABS Spheres
                    sdfType = 2;
                }
            }
            if (sdfType == 4) sdfType = 5;
            if (sdfType == 3) sdfType = 4;
            if (sdfType == 2 && math.abs(lossyScale.x - lossyScale.y) < DELTA_SCALE && math.abs(lossyScale.y - lossyScale.z) < DELTA_SCALE) sdfType = 3;

            var rot = _sdfObjs[i].transform.rotation * extraRot;
            if (localSetup && _sdfObjs[i].parent != null)
            {
                _collidableSDFs[i].center = _sdfObjs[i].parent.InverseTransformPoint(_collidableSDFs[i].center);
                rot = Quaternion.Inverse(_sdfObjs[i].parent.rotation) * rot;
            }

            _collidableSDFs[i].extent = extent;
            _collidableSDFs[i].rotation = QuatToVec(rot);
            _collidableSDFs[i].sdfType = sdfType;
            //print("sdfType " + sdfType);
        }

        private float GetWindTurbulence(float time, float windPulseFrequency, float windPulseMagnitude)
        {
            return Mathf.Clamp(Mathf.Clamp01(Mathf.PerlinNoise(time * windPulseFrequency, 0.0f)) * 2 - 1, -1, 1) * windPulseMagnitude;
        }

        public Vector4 QuatToVec(Quaternion rot)
        {
            Vector4 rotVec;
            rotVec.x = rot.x;
            rotVec.y = rot.y;
            rotVec.z = rot.z;
            rotVec.w = rot.w;
            return rotVec;
        }

        private Vector3 QuatToVec3(Quaternion q)
        {
            Vector3 v;
            v.x = q.x;
            v.y = q.y;
            v.z = q.z;
            return v;
        }

        private Vector3 DiffPos(Vector3 position, Vector3 prevPosition, float dt)
        {
            return (position - prevPosition) / dt;
        }

        private Vector3 DiffRot(Quaternion rotation, Quaternion prevRotation, float dt)
        {
            return QuatToVec3(rotation * Quaternion.Inverse(prevRotation)) * 2.0f / dt;
        }

        private int InitializeDiffFrames(Vector3 position, Vector3 scale, Quaternion rotation)
        {
            TransformDynamics td = new TransformDynamics();
            td.frame = new TransformDynamics.TransformPerFrame(position, rotation, scale);
            td.prevFrame = td.frame;
            td.velocity = Vector3.zero;
            td.rotVelocity = Vector3.zero;
            td.posAcceleration = Vector3.zero;
            td.rotAcceleration = Vector3.zero;
            _bodyTds.Add(td);
            _bodyTdsCount++;
            return _bodyTdsCount - 1;
        }

        private int InitializeDiffFrames(Vector3 position, Vector3 scale, Quaternion rotation, Transform parent, bool localSetup = false)
        {
            TransformDynamics td = new TransformDynamics();
            if (localSetup && parent != null)
            {
                position = parent.InverseTransformPoint(position);
                rotation = Quaternion.Inverse(parent.rotation) * rotation;
            }
            td.frame = new TransformDynamics.TransformPerFrame(position, rotation, scale);
            td.prevFrame = td.frame;
            td.velocity = Vector3.zero;
            td.rotVelocity = Vector3.zero;
            td.posAcceleration = Vector3.zero;
            td.rotAcceleration = Vector3.zero;
            _tds.Add(td);
            return _tds.Count - 1;
        }

        private void UpdateDiffFrames(Transform body, float dt, int tdNum = 0)
        {
            if (_localSpace)
            {
                UpdateDiffFrames(body, dt, tdNum, out Vector4 posVel, out Vector4 rotVel);
                //m_cs.SetVector("_posCloth", body.position);
                posVel = Quaternion.Inverse(body.rotation) * posVel;
                posVel.w = _worldPositionImpact;
                _cs.SetVector("_posVel", posVel);
                rotVel.w = _worldRotationImpact;
                _cs.SetVector("_rotVel", rotVel);
                //Debug.DrawLine(body.position, body.position + (Vector3)posVel * posVel.w);
            }
            else
            {
                _cs.SetVector("_posVel", Vector4.zero);
                _cs.SetVector("_rotVel", Vector4.zero);
            }
        }

        private void UpdateDiffFrames(Transform t, float dt, int tdNum, out Vector4 posVel, out Vector4 rotVel)
        {
            var td = _bodyTds[tdNum];

            Vector3 position = t.position;
            Quaternion rotation = t.rotation;
            Vector3 scale = t.lossyScale;

            td.prevFrame = td.frame;
            td.frame.position = position;
            td.frame.rotation = rotation;
            td.frame.scale = scale;

            td.velocity = DiffPos(td.frame.position, td.prevFrame.position, dt);
            td.rotVelocity = DiffRot(td.frame.rotation, td.prevFrame.rotation, dt);

            //Debug.Log(t.name + " velocity:" + td.velocity.x + ", " + td.velocity.y + ", " + td.velocity.z);
            //Debug.Log(t.name + " rotVelocity:" + td.rotVelocity.x + ", " + td.rotVelocity.y + ", " + td.rotVelocity.z);

            posVel = td.velocity;
            rotVel = td.rotVelocity;

            _bodyTds[tdNum] = td;
        }

        private void UpdateDiffFrames(Transform t, Transform parent, float dt, int tdNum, out Vector3 posVel, out Vector3 rotVel, bool localSetup = false)
        {
            var td = _tds[tdNum];

            Vector3 position = t.position;
            Quaternion rotation = t.rotation;
            if (localSetup && parent != null)
            {
                position = parent.InverseTransformPoint(position);
                rotation = Quaternion.Inverse(parent.rotation) * rotation;
            }
            Vector3 scale = t.lossyScale;

            td.prevFrame = td.frame;
            td.frame.position = position;
            td.frame.rotation = rotation;
            td.frame.scale = scale;

            td.velocity = DiffPos(td.frame.position, td.prevFrame.position, dt);
            td.rotVelocity = DiffRot(td.frame.rotation, td.prevFrame.rotation, dt);

            //Debug.Log(t.name + " velocity:" + td.velocity.x + ", " + td.velocity.y + ", " + td.velocity.z);
            //Debug.Log(t.name + " rotVelocity:" + td.rotVelocity.x + ", " + td.rotVelocity.y + ", " + td.rotVelocity.z);

            posVel = td.velocity;
            rotVel = td.rotVelocity;

            _tds[tdNum] = td;
        }


        public void OnDrawGizmos()
        {
            if (_dynamics == null || _positions == null)
            {
                return;
            }

            var posData = new Vector3[_positions.count];
            _positions.GetData(posData);

            //var tex = (RenderTexture)_sdfTex.sdf;
            //float3 gridSize = new float3(tex.width, tex.height, tex.volumeDepth);
            //float3 bCenter = _sdfTex.transform.position;
            //float3 bSize = _sdfTex.size;
            //float3 scale = bSize / gridSize;

            //ComputeBuffer sdfDebug = new ComputeBuffer(_positions.count, sizeof(float)*3);

            //int kernel = 21;
            //_cs.SetInt("params_numParticles", _simParams.numParticles);
            //_cs.SetFloat("params_friction", _dynamics._globalSimParams.friction);
            //_cs.SetBuffer(kernel, "predicted", _predicted);
            //_cs.SetBuffer(kernel, "positions", _positions);
            //_cs.SetFloat("deltaTime", Time.deltaTime);

            //_cs.SetVector("_bCenter", (Vector3)bCenter);
            //_cs.SetVector("_bSize", (Vector3)bSize);
            //_cs.SetVector("_scale", (Vector3)scale);
            //_cs.SetVector("_gridSize", (Vector3)gridSize);
            //_cs.SetTexture(kernel, "_sdfVoxelData", tex);
            //_cs.SetBuffer(kernel, "_sdfDebug", sdfDebug);
            //_cs.Dispatch(kernel, _simParams.numParticles.GetComputeShaderThreads(BLOCK_SIZE), 1, 1);

            //var sdfData = new Vector3[sdfDebug.count];
            //sdfDebug.GetData(sdfData);

            //sdfDebug.Release();


            Gizmos.color = Color.white;

            if (_debugPoints)
            {
                int mCount = _meshDataList.Count;
                Color[] colors = new Color[mCount];
                for (int n = 0; n < mCount; n++)
                {
                    colors[n] = _meshDataList[n].go.GetComponent<Renderer>().material.color;
                }

                for (int i = 0; i < posData.Length; i++)
                {
                    for (int n = 0; n < mCount; n++)
                    {
                        var mData = _meshDataList[n];
                        if (i >= mData.prevNumParticles && i < mData.prevNumParticles + mData.newParticles)
                        {
                            Gizmos.color = colors[n];
                            break;
                        }
                    }
                    Gizmos.DrawSphere(posData[i], _dynamics._globalSimParams.particleDiameter * 0.05f);

                    //Gizmos.DrawLine(posData[i], posData[i] + sdfData[i]);
                }
            }
            if (_mouseGrabber != null)
            {
                Gizmos.color = Color.red;
                var handleData = new MouseGrabberGPU.HandleMouse[1];
                _mouseGrabber._handleBuffer.GetData(handleData);
                int index = handleData[0].objectIndex;
                if (index >= 0 && index < posData.Length) Gizmos.DrawSphere(posData[index], _dynamics._globalSimParams.particleDiameter * 1);
            }

            //if (_debugShowNeighbors && m_spatialHash != null && _collectSpheres != null)
            //{
            //    var neighbors = new int[m_spatialHash.neighbors.count];
            //    m_spatialHash.neighbors.GetData(neighbors);

            //    var sphereData = new float4[_collectSpheres._sphereDataBuffer.count * 3];
            //    _collectSpheres._sphereDataBuffer.GetData(sphereData);

            //    int params_maxNumNeighbors = m_dynamics._globalSimParams.maxNumNeighbors;
            //    int params_numObjects = m_spatialHash._maxNumObjects;
            //    int params_numParticles = _simParams.numParticles;
            //    int id = _getNeighborsOfId;
            //    Gizmos.color = Color.white;
            //    Gizmos.DrawSphere(posData[id], m_dynamics._globalSimParams.particleDiameter * 0.5f);
            //    for (int neighbor = id; neighbor < params_numObjects * params_maxNumNeighbors; neighbor += params_numObjects)
            //    {
            //        int j = neighbors[neighbor];
            //        int nj = j - params_numParticles;
            //        if (j >= 1 && j < params_numParticles)
            //        {
            //            Gizmos.color = Color.red;
            //            Gizmos.DrawSphere(posData[j], m_dynamics._globalSimParams.particleDiameter * 0.5f);
            //        }
            //        else if (nj >= 0 && nj < _collectSpheres._sphereDataBuffer.count)
            //        {
            //            var pos = sphereData[nj * 3].xyz;
            //            Gizmos.color = Color.green;
            //            Gizmos.DrawSphere(pos, m_dynamics._globalSimParams.particleDiameter * 0.5f);
            //        }
            //    }
            //}
        }

        public const int BLOCK_SIZE = 256;
        public enum UpdateModes
        {
            FixedUpdate,
            Update,
            LateUpdate,
            OnAnimatorMove
        }
        [Tooltip("You should always use Fixed Update! You can try OnAnimatorMove if you have an Animator, you need to link it to make it work (See AnimatorController script).")]
        public UpdateModes _updateMode = UpdateModes.FixedUpdate;

        [Tooltip("This will run the sim, should always be on. You can use it to pause/restart the simulation.")]
        public bool _runSim = true;
        [Tooltip("You can use the mouse to grab a cloth-particle by clicking the left mouse button.")]
        public bool _useMouseGrabber = false;
        [Tooltip("This let you toggle between triangle and vertex collision for mesh colliders.")]
        public bool _trisMode = true;
        [Tooltip("This tranforms all clothes to the local space of their body parent.")]
        public bool _localSpace = false;
        [Range(0.0f, 10.0f)]
        [Tooltip("This is the global impact on the particles when the parent object changes the position.")]
        [SerializeField] public float _worldPositionImpact = 1.0f;
        [Range(0.0f, 10.0f)]
        [Tooltip("This is the global impact on the particles when the parent object changes the rotation.")]
        [SerializeField] public float _worldRotationImpact = 1.0f;

        [Tooltip("If you turn this on, you can use the y-scaling of your plane mesh to increase the bias between plane and cloth.")]
        /*[SerializeField] */
        private bool _usePlaneScaleY = false;

        [Tooltip("This will show you all particles of the cloth objects as Gizmos! This will affect performance!")]
        public bool _debugPoints = false;

        //public bool _useExtraParticleCollisionStep = true;
        //private int _loopCount = 1;
        [Tooltip("If active, you have to apply all the objects by yourself.")]
        public bool _manualSetup = true;
        [Tooltip("These objects will be ignored by the sim, will affect all sdf colliders!")]
        public GameObject[] _ignoreObjects;
        //[Tooltip("Automatically collects all colliders in the scene and adds them here!")]
        //public bool _autoCollect = false;
        [Tooltip("List of all Unity colliders for this sim, will be added automatically!")]
        public Collider[] _unityColliders;
        [Tooltip("List of all Automatic-Bone-Spheres objects for this sim, can be added via the ABS script.")]
        public GameObject[] _absColliders;

        RenderTexture _dummyTex;

        public CollisionMeshesGPU _collisionMeshes;

        List<ClothSkinningGPU> _listOfClothSkinners = new List<ClothSkinningGPU>();

        private const float DELTA_SCALE = 0.001f;

        internal ComputeBuffer _positions;
        //internal ComputeBuffer positions2;
        internal ComputeBuffer _normals;
        ComputeBuffer _indices;

        internal ComputeBuffer _velocities;
        internal ComputeBuffer _predicted;
        ComputeBuffer _deltas;
        ComputeBuffer _deltaCounts;
        internal ComputeBuffer _invMasses;

        ComputeBuffer _stretchIndices;
        ComputeBuffer _stretchLengths;
        //ComputeBuffer _bendIndices;
        //ComputeBuffer _bendAngles;

        //// Attach attachParticleIndices[i] with attachSlotIndices[i] w
        //// where their expected distance is attachDistances[i]
        ComputeBuffer _attachParticleIDs;
        ComputeBuffer _attachSlotIDs;
        ComputeBuffer _attachDistances;
        ComputeBuffer _attachSlotPositions;

        internal List<float3> _attachSlotPositionsData = new List<float3>();
        List<int> _attachParticleIDsData = new List<int>();
        List<int> _stretchIndicesData = new List<int>();
        List<int> _attachSlotIDsData = new List<int>();
        List<float> _stretchLengthsData = new List<float>();
        List<float> _attachDistancesData = new List<float>();
        List<float> _invMassesData = new List<float>();
        public GPUClothDynamicsV2 _dynamics;

        internal SpatialHashGPU _spatialHash;

        MouseGrabberGPU _mouseGrabber;

        public struct SimParams
        {
            public int numParticles;
            public int numSubsteps;
            public int numIterations;
            public float maxSpeed;
            public float particleDiameter;
            public float deltaTime;
        }
        public SimParams _simParams;

        private ComputeShader _cs;

        int _numCollidableSDFs;
        CollidableSDFStruct[] _collidableSDFs;

        internal ComputeBuffer _collidableSDFsBuffer = null;
        class SDFObject
        {
            internal Transform transform;
            internal Transform parent;
        }
        private List<SDFObject> _sdfObjs;

        public class MeshData
        {
            public GameObject go;
            public int id;
            public int newParticles;
            public int prevNumParticles;
            public float4x4 modelMatrix;
        }
        public List<MeshData> _meshDataList = new List<MeshData>();
        List<Vector3> _positionsData = new List<Vector3>();
        List<Vector4> _velocitiesData = new List<Vector4>();
        List<Vector3> _normalsData = new List<Vector3>();
        List<int> _indicesData = new List<int>();

        private List<TransformDynamics> _bodyTds = new List<TransformDynamics>();
        private int _bodyTdsCount = 0;

        class ConnectedData
        {
            internal Transform transform;
            internal float3 offset;
        }

        List<ConnectedData> _connectedObjects = new List<ConnectedData>();

        struct CollidableSDFStruct
        {
            public Vector3 center;
            public Vector4 extent;
            public Vector4 rotation;
            public Vector3 posVel;
            public Vector3 rotVel;
            public int sdfType;
        }

        private struct TransformDynamics
        {
            public struct TransformPerFrame
            {
                public Vector3 position;
                public Quaternion rotation;
                public Vector3 scale;
                public TransformPerFrame(Vector3 position, Quaternion rotation, Vector3 scale)
                {
                    this.position = position;
                    this.rotation = rotation;
                    this.scale = scale;
                }
            }
            public TransformPerFrame frame;
            public TransformPerFrame prevFrame;
            public Vector3 velocity;
            public Vector3 rotVelocity;
            public Vector3 posAcceleration;
            public Vector3 rotAcceleration;
        }
        private List<TransformDynamics> _tds = new List<TransformDynamics>();
        private global::Unity.Collections.NativeArray<CollidableSDFStruct> _sdfBuffer;

    }

    public static class ListExtra
    {
        public static void Resize<T>(this List<T> list, int sz, T c)
        {
            int cur = list.Count;
            if (sz < cur)
                list.RemoveRange(sz, cur - sz);
            else if (sz > cur)
            {
                if (sz > list.Capacity)//this bit is purely an optimisation, to avoid multiple automatic capacity changes.
                    list.Capacity = sz;
                list.AddRange(Enumerable.Repeat(c, sz - cur));
            }
        }
        public static void Resize<T>(this List<T> list, int sz) where T : new()
        {
            Resize(list, sz, new T());
        }
    }
}
